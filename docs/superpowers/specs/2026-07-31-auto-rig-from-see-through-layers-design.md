# See-Through 图层自动绑骨设计（Auto-Rig）

## Status

**Revision 13，把 join 提升为真正的终态 G stage、补齐公共 per-item failure artifact，并让 C 单次编码 canonical PNG、D/E 只复制字节；正式交付仍为 Spine 4.2 + Live2D runtime。未动生产代码。**

本文覆盖三件事：

1. 对参考项目 `MangoLion/stretchystudio`（commit
   `24a83a27ba43e43e9d2e3de5e33994594e6199c2`，2026-07-31 抓取）绑骨流水线的评审结论；
2. 本仓库在 `module/see_through/` 产物之上新增自动绑骨阶段的设计契约；
3. 面向批量生产，直接导出带能力匹配的预设动作/表情的 Spine 4.2 包和 Live2D Cubism
   runtime 包，不建设预览产品。

评审基准：

- `src/io/armatureOrganizer.js`（骨架推断 + DWPose 推理，27 KB）
- `src/io/splitLR.js`（客户端左右拆分）
- `src/mesh/generate.js`、`src/mesh/contour.js`、`src/mesh/sample.js`
- `src/io/live2d/bodyAnalyzer.js`（后期的掩膜测量实现）
- `src/io/live2d/moc3writer.js`、`model3json.js`、`motion3json.js`、`cdi3json.js`
- `src/io/live2d/cmo3/deformerEmit.js`、`docs/live2d-export/MOC3_FORMAT.md`、
  `docs/live2d-export/WARP_DEFORMERS.md`（deformer 层级、局部坐标与 MOC3 section 证据）
- `src/io/exportSpine.js`、`docs/elbow_implementation.md`

本仓库基准：

- `module/see_through/vendor/utils/inference_utils.py`
- `module/see_through/extracted/postprocess_core.py`
- `module/see_through/runner.py`、`module/onnx_runtime/session.py`
- `config/model.toml:248`、`gui/wizard/step6_tools.py:233`

---

## Linus Review：参考项目的绑骨流水线

### 三个前置问题

**这是真问题吗？** 是。see-through 已经把插画拆成语义图层，但图层本身不能动。
从图层到可驱动角色之间缺的是关节位置、父子关系、网格和权重。这四类输出在输入信息
充分时可通过约束求解；遮挡或 merged 部件造成信息不足时，正确结果是显式判为不可解，
而不是伪造一套看似完整的 Rig。

**有更简单的办法吗？** 有，而且参考项目自己没走。它只用图层 **bbox** 猜关节，
但图层本身带着**逐像素 alpha 掩膜**；本仓库的 see-through 还额外产出**逐部件深度图**
（`final_depth.psd`、`<tag>_depth.png`、`info.json` 里的 `depth_median`，见
`module/see_through/vendor/utils/inference_utils.py:315-354`、`:521-534`）。
对已经正确拆分的肢体，用掩膜测量得到的几何约束通常比 bbox 中点更强，而且不需要
神经网络。参考项目后来自己也承认了这一点——`src/io/live2d/bodyAnalyzer.js` 就是
"改用掩膜逐行测量"的第二版实现，
但它只服务 Live2D 导出，**没有回灌骨架推断**，两套逻辑各写各的。

**会破坏什么？** 在本仓库里新增一个独立阶段不破坏任何现有行为，
前提是不去改 see-through 的产物格式、不改 `tblr_split` 默认值。
真正的风险是：一旦绑骨消费 `optimized/info.json`，这个原本的内部中间产物就变成了
对外 API，以后 see-through 升级会连带打断绑骨。这条必须用固定 fixture 钉死。

### 结论

参考项目的**流程骨架是对的**（拆图层 → 语义匹配 → 关节 → 骨骼树 → 网格权重 → 导出），
**具体实现有明确优化空间**，而且有几处是真 bug，不是风格问题。逐条列在下面，
按严重程度排序。

本仓库不复制参考项目的状态模型。以下不变量是设计门槛：

1. 一个版本化 `RigDocument` 是唯一事实源；阶段文件只是可失效的缓存或诊断产物；
2. 姿态模型只提交观测，不创建或重建骨骼；
3. 诊断、动作绑定与导出读取同一份 Rig，不允许导出阶段再次“自动绑骨”；
4. 导出器只做坐标和格式编码，不能静默丢权重或修补缺失关节；
5. 任何无法可靠求解的关节必须标记 `unresolved` 或显式降级，不能用画布中心伪造成功；
6. resume 必须验证输入、配置、算法版本和产物摘要，不能只看文件是否存在。

---

### F1（正确性）左右约定在参考项目内部自相矛盾

`armatureOrganizer.js:169-170` 有 topwear 时：

```js
kp.lShoulder = { x: topwear.x + topwear.w * 0.85, ... }   // l = 大 x
kp.rShoulder = { x: topwear.x + topwear.w * 0.15, ... }   // r = 小 x
```

`armatureOrganizer.js:177-178` 无 topwear 的 fallback 分支：

```js
kp.lShoulder = { x: face.cx - face.w * 0.2, ... }         // l = 小 x
kp.rShoulder = { x: face.cx + face.w * 0.2, ... }         // r = 大 x
```

同一个函数里两个分支的左右**正好反过来**。半身像（无 topwear）走 fallback 分支，
于是半身像的手臂骨骼是镜像错的。

`splitLR.js` 更明显：文件头注释（第 10-13 行）写"较小 centroid X 判为 left，
这与 armatureOrganizer 里 'l' 在较小 X 的约定一致"，而代码（第 127-129 行）写的是
"较小 centroid X → 角色的右侧（-r）"。注释和代码互相打脸，而且注释说的
"armatureOrganizer 的约定"根本不存在。

**本仓库上游的约定是明确的**：`inference_utils.py:300-312` 的 `label_lr_split`
先返回中心 x 较小的连通域，`part_lr_split`（`:380-399`）把它写成 `<tag>-r`。
即 **`-r` = 图像较小 x = 观众左侧 = 角色右侧**，与 COCO/DWPose 的 left/right 语义一致。

**修法**：内部数据结构里**不要出现 left/right**，只存 `side: "xmin" | "xmax" | null`
（图像空间事实）。`-l`/`-r` 只在读 tag 和写导出文件这两处做一次映射，映射表放一个地方，
配一个 fixture 测试钉住。解剖学左右只有在拿到朝向线索（正面/背面）时才有意义，
而两个项目都没有做朝向判断——背面视角下这套标签整体反转，这一点必须写进文档而不是
装作不存在。

### F2（正确性）bbox 启发式的肘/膝位置在几何上就是错的

`armatureOrganizer.js:189`：

```js
kp.lElbow = { x: (kp.lShoulder.x + kp.lWrist.x) / 2, y: (...) / 2 };
```

肘 = 肩腕中点。这只在手臂完全伸直时成立。插画里手臂几乎总是弯的，
中点会落在手臂轮廓**外面**——即肘关节枢轴在图层的透明区域里，绕它旋转必然撕裂。

腕的位置同样可疑：`:188` 取 `handwear` bbox 的 `y + h*0.1`（顶部偏上中心），
这假设手臂朝下垂。举手、抱胸、叉腰全错。

**修法**：手臂/腿图层的 alpha 掩膜就在手里，直接量：

1. 对部件掩膜做连通域清理和小孔闭合，再求中轴；
2. 把中轴变成 8 邻接图，按局部 distance-transform 半径和支路长度剪掉手指、衣褶产生的短刺；
3. 用与躯干掩膜的接触区确定近端，在剩余图上取近端到最远端的主路径；
4. 肘/膝只在主路径中间区间内搜索，以平滑后曲率峰和两段折线拟合误差共同打分；
5. 近乎直线、曲率峰不显著时退化为**弧长中点**，不是肩腕欧氏中点；
6. 关节点必须落在掩膜内，落不进去就沿主路径投影回去。

只取“全骨架曲率最大点”会稳定选中手指、袖口或 mask 毛刺，不可接受。主路径同时是 F7
权重方案的输入，不是额外开销。

### F3（正确性）姿态推理跑在错误的图像上

`armatureOrganizer.js:286-296` 把所有**已经 inpaint 过的**图层按数组顺序重新合成，
底色填黑，然后送去推理。这个合成图和原始插画不是同一张：amodal 补全出来的
被遮挡部分会盖到前面的部件上，纯黑底还会让 RGBA 边缘变暗。

本仓库根本不需要重新合成——`item_dir/src_img.png` 就是 LayerDiff 实际消费并与所有 mask
共坐标系的方形 letterbox 图（`module/see_through/runner.py:277-278` 拿它做 resume 判据）。
它不是原始文件分辨率下的图像；这一区别在输入契约中单独冻结。

同一函数的 `:299-310` 把**整张画布**做 letterbox 当成人体框喂给 top-down 模型。
top-down 模型的精度直接取决于人体框的紧致度；画布留白越多，有效分辨率越低。
应该用所有身体部件 alpha 的并集 bbox，按模型宽高比外扩后做仿射，
这是 top-down 姿态估计的标准做法。

### F4（正确性）置信度算了但扔掉了

`armatureOrganizer.js:354-358` 算出 `conf: Math.min(mx, my)` 之后，
`applyDWPoseKeypoints`（`:368-427`）从头到尾没读过它，只做 clamp 到画面内。
模型在动漫图上把手腕预测到画面外，clamp 之后变成一个贴边的"合法"关节点，
下游无从分辨。

模型虽然输出 133 点，参考代码只映射前 17 个 body 点；后续骨骼 pivot 实际只直接消费
鼻、双眼、双肩、双肘、双髋、双膝这 11 个原始点。耳、手腕和脚踝已经解码却被骨骼拓扑
丢掉，链条停在肘/膝。这不是模型能力问题，是中间数据结构先把信息截断了。

**修法**：关节点必须带 `confidence`，低于阈值的直接判为缺失，交给几何求解；
另外加一条硬校验——关节点必须落在它所属部件的掩膜内（或掩膜膨胀 N px 内），
不满足就拒绝。这条校验掩膜里现成，成本接近零。

### F5（结构）两条互不相干的估计分支 + 兜底堆

`estimateSkeletonFromBounds`（bbox 启发式）和 `runDWPose` 是两个完全独立的函数，
各自产出同一个 `kp` 字典，调用方二选一。结果就是 `:224-236` 那一坨：

```js
if (!kp.pelvis)      kp.pelvis      = { x: cx, y: cy };
if (!kp.neck)        kp.neck        = { ... };
... 共 11 行
```

11 行纯粹为了补齐"另一条分支可能没填的字段"。`buildArmatureNodes:467-481` 里还有
第二处补丁——DWPose 分支不产 `headBase`，于是在骨骼构建函数里又重算了一遍 bbox。
数据结构没设计好，就得靠分支打补丁。

**修法**：只有一条管线。几何求解**永远执行**并产出关节观测；姿态模型（如果开启）
产出另一组先验观测。geometry 也可能因粘连/断裂而 unresolved，不宣称永远产出完整关节；
不同来源的 confidence 未校准时也不能直接加权。所有 observations 先收集、校验、决策，
最后只物化一次 Rig。部件缺失时对应骨骼不创建，求解失败时保留 unresolved，禁止塞画布
中心假坐标，也禁止先创建 bbox Rig 再删除并替换成 pose Rig。

### F6（结构）一张骨骼表被拆成四份手工同步的字典

`armatureOrganizer.js:486-543` 是四个平行结构：`needGroup`（15 项）、`pivots`（15 项）、
`parentBone`（15 项）、`CREATE_ORDER`（15 项手写顺序数组），加上 `:591-606` 的
`SKELETON_CONNECTIONS`——第五份，内容是 `parentBone` 的重复表达。加一根骨头要改五处。
`boneForTag(tag, groups)`（`:436-449`）签名里的 `groups` 参数从头到尾没被用过。

**修法**：一张声明式骨骼表，每条记录 `{name, parent, pivot_from, requires}`。
创建顺序由 `parent` 拓扑排序得出，连线由 `parent` 直接得出，
`needGroup` 由 `requires` 对现有 tag 求值得出。五份变一份。

### F7（正确性）蒙皮权重用了一个和分辨率绑死的常量

`docs/elbow_implementation.md` 记载的权重公式：

```js
const proj   = (v.x - jx) * axX + (v.y - jy) * axY;
const weight = clamp(proj / 40 + 0.5, 0, 1);
```

`40` 是像素常量。2048 px 的插画上 40 px 的过渡带等于没有过渡（硬切），
512 px 上 40 px 覆盖大半条胳膊。而且这是沿**直线**投影，参考项目自己的文档也标了
`Linear Projection Bias`——弯曲手臂上权重会串到不该串的地方（掌心和上臂在直线投影上
可能距离很近，但沿手臂走是两端）。

**修法**：权重沿 F2 求出的**中轴弧长**参数化，不沿直线投影；过渡带半宽取
关节处局部肢体宽度的固定倍数（distance transform 在关节点的值直接给出半宽），
彻底去掉像素常量。弯曲肢体自动正确，因为弧长走的是掩膜内部。

### F8（正确性）Spine mesh 编码本身不合法，并且把权重丢了

`exportSpine.js:181-185`：

```js
if (part.mesh) {
  attachment.type = "mesh";
  attachment.vertices = part.mesh.vertices;
  ...
}
```

不只是没权重。`part.mesh.vertices` 在参考项目内部是对象数组，`triangles` 是三元组数组，
代码却原样写入 Spine JSON。Spine 要求二者都是扁平数值数组。即使刚性 mesh 也不满足
目标格式；编辑器里算出的 `boneWeights` 和 `jointBoneId` 同样完全没有编码。

**修法**：导出必须有独立 `SpineMeshEncoder`，不能把内部对象 `JSON.stringify`：

- 未加权顶点：`[x0, y0, x1, y1, ...]`；
- 加权顶点：每个顶点编码为
  `boneCount, (boneIndex, bindX, bindY, weight) × boneCount`；
- `triangles`、`uvs` 必须扁平；
- 顶点坐标必须转换到对应骨骼的 bind-local 空间；
- `hull`、图集 region 和 attachment path 必须一致。

这是 Spine 官方 JSON 契约，已核对，不再列为待确认。动作只使用骨骼 timeline 时可以没有
deform timeline；一旦表情 preset 使用顶点形变，就必须从 `MotionClip` 明确编码 deform，
不能在 exporter 内临时生成。无论是否有动画，丢掉静态蒙皮权重都是 bug。

### F9（性能/正确性）网格生成的两个老问题

`mesh/generate.js:110-121` 去重是 O(n²)：每个点线性扫描已保留点。
`numEdgePoints=80` 时无所谓，网格密一点就爆。换网格哈希即可 O(n)。

`mesh/generate.js:124` 对点云做**无约束** Delaunay。凹形部件（C 形头发、
张开的手指、镂空饰品）会被凸包填实，多出来的三角形横跨透明区，
形变时会拽出可见的膜。参考项目靠 `dilateAlphaMask` 膨胀 2 px 缓解，
这只对细缝有效，对大凹口无效。

**修法**：不同连通分量分别三角化，禁止跨分量连边；三角化后按掩膜剔除——重心或
任一边的多点采样落在透明区就丢弃；再按相对采样间距归一化的外接圆半径剔除狭长片。
比首版引入约束 Delaunay 库简单，且不会把两个分离发束直接缝起来。

### F10（正确性）不能假设 see-through 已经解决四肢左右拆分

`splitLR.js` 用 8 连通域 + 取最大两块来拆左右。手臂贴住身体、两只手交叠时
连通域只有一块，直接失败（`:113-116` 返回 null）。

本仓库的 `part_lr_split`（`inference_utils.py:380-399`）对左右部件同样只是连通域。
更精确地说，当前 v3 管线中 `cluster_inpaint_part` **完全不会执行**：Marigold 已把 `hair`
compose 成 `front hair` / `back hair` 后 `continue`，根 `info.json` 不会出现 `hair`；眼部同理，
不会出现根 tag `eyes`。`further_extr` 的 hair 分支还被第二层 `if tblr_split` 包住。
因此 v3 的前后发来自 LayerDiff 直接输出，不存在 `hairf/hairb`，也没有任何深度聚类处理手臂
或腿。当前 `tblr_split=true` 只尝试拆 `handwear`、`eyewhite`、`irides`、`eyelash`、
`eyebrow`、`ears`，不拆 `legwear` 或 `footwear`。所以“上游的深度聚类已经解决四肢粘连”
这个前提是错的。

**修法**：auto-rig 自己做明确的输入分级：

1. 已有 `-l/-r` 标签：直接使用；
2. merged mask 有两个可靠连通域：按 `xmin/xmax` 拆分；
3. 连通域粘连或只分出一侧：标记 `merged_limb` / `partial_limb`，首版不伪造双侧骨骼；
4. 姿态先验可以提供诊断和候选种子，但不能凭 17 个点把一张粘连纹理安全切成两层。

注意 `config/model.toml:262` 当前默认 `tblr_split = false`。**不要改这个默认值**
（会改变现有用户的 see-through 产物）。绑骨阶段检测到未拆分四肢时，默认把对应层
刚性挂到 torso/root，并写入结构化诊断；“单骨双臂”会制造一个看似可动但语义错误的 rig，
不采用。

### F11（信息丢失）参考项目完全没用深度

`armatureOrganizer.js` 全文没有 depth 相关逻辑。它只能读 PSD，深度信息在 PSD 里
只体现为图层顺序。本仓库有 `final_depth.psd` + 每部件 `depth_median`，可以用于：

- 绘制顺序 / Spine slot 顺序（已有，但目前只是 PSD 顺序的副产物）；
- 前后手臂判定：两条 `handwear-*` 的 `depth_median` 直接给出谁在前，
  这比任何 2D 姿态模型都可靠；
- merged/crossing 部件的候选排序与遮挡诊断。正常蒙皮先按语义限制候选骨骼，
  不让所有空间邻近骨骼竞争，因此不再额外发明一套“深度门控权重”。

这是本仓库相对参考项目的**结构性优势**，设计里必须用上。

### F12（范围）Live2D runtime 可以纳入交付，但 `.cmo3` 不能顺手捎上

“Live2D 格式”实际包含两类不同交付物。可由 SDK 直接加载的是 runtime bundle：`.moc3`、
`.model3.json`、纹理、`.motion3.json`、`.exp3.json` 等；可由 Cubism Editor 继续编辑的是
`.cmo3/.can3` 工程。后者是未公开且带封装/混淆的私有工程格式。参考项目为此做了 30+ 个
session，`cmo3writer.js` 单文件约 230 KB；而 runtime 的 `moc3writer.js` 约 42 KB。

因此正式交付纳入 **SDK 可加载的 Live2D Cubism runtime bundle**，但不承诺可编辑
`.cmo3/.can3`。这不是降低 Live2D 交付要求，而是拒绝把“运行格式”和“编辑器工程”混成一项。
参考项目的 runtime writer 也不能直接当成成品。虽然它列出了 RotationDeformer 的二进制
sections，实际 `buildSectionData` 只收集 WarpDeformer，并把总 deformer 数设为 warp 数，没有
写出 RotationDeformer bone hierarchy；它也没有完整的 LBS 混合区运行时编译和 `.exp3.json`
导出。上述能力都是本需求“带动作和表情”的硬条件，必须在本项目补齐并通过官方 runtime
验证，不能再把“所有骨骼都烘焙成 ArtMesh 顶点”当作捷径。

### F13（结构）预览 Rig 与 Live2D 导出 Rig 有两个事实源

参考项目在 `CanvasViewport.jsx` 的 `WARP_SPECS` 里生成编辑器参数和 warp，
`cmo3writer.js` 又按自己的硬编码规则重新分析图层并生成另一套 deformers/keyforms。
仓库自己的 `WARP_EXPORT_AUDIT.md` 已记录覆盖差异。具体缺几个参数并不重要：只要两套表
需要人工同步，预览与导出就必然漂移。

**本设计的约束**：`RigDocument` 是唯一事实源。几何求解、人工 override 和参数生成只在
构建 Rig 时发生一次。所有导出器只读这个文档，禁止在 exporter 内重新判断骨骼或权重。

### F14（数据丢失）参考项目保存工程会遗漏 `physicsRules`

参考项目的 project store 和 Live2D exporter 都使用 `physicsRules`，但 `.stretch` 保存器只
写了 `physics_groups`。保存再打开会丢失用户编辑过的物理规则。这不是本设计首版的功能，
却说明“内存对象随手 JSON 化”不是可靠契约。

**本设计的约束**：公共 Rig schema、缓存 schema 和导出 schema 分开定义；保存/加载必须
做完整往返测试。未来增加字段必须升级 `schema_version` 并提供迁移，不接受在加载器各处
散落 `if field is None` 补丁。

### F15（兼容性）导出名称不是显示名称清洗一下就完事

参考 Spine exporter 删除所有非 ASCII 字符；中文名可能变成空字符串，不同名称也可能清洗
成同一个 ID。父子关系和动画引用再用这些名称连接，碰撞后就是静默串骨。

**本设计的约束**：内部引用只用稳定 ID。C 阶段从完整 Rig、preset/parameter registry 和所有
格式的派生实体键一次性构造 `GlobalExportSymbolTable`，写入 `RigDocument`；两个 exporter 只能取
子集，不能各自维护“已占用名称”或临时调用 `sanitize_name()`。ASCII slug 为空、冲突或碰到保留的
Live2D standard parameter ID 时，只在相同 canonical `ExportNamespaceKey` 内为非保留项附加稳定短 ID；
不同格式/section/局部 owner scope 不是一个扁平名称池。映射与摘要写入两份导出报告；剪枝、optional
capability 和 exporter 执行顺序都不得改变同一 typed key 的 namespace/name，跨格式对应实体仍共享
`base_export_name`。

### F16（验证）复杂格式没有测试就不算支持

参考项目没有 `test` script，却手写了数千行 CMO3/MOC3 和 Spine 编码。静态生成一个 ZIP
不等于目标软件能读。本设计要求纯结构验证、golden fixture 和目标软件/运行时验收三层
测试；无法在普通 CI 运行的商业软件验收必须做成显式 opt-in gate，不能用“手工试过一次”
代替。

---

## 设计

### Revision 3 输入契约审计结论

| 审查项 | 代码核对结果 | 设计处理 |
|---|---|---|
| R1 canvas 语义 | **成立**。`src_img.png` 是 `center_square_pad_resize` 后的 `resolution×resolution` 方形图 | Rig v1 全程使用 letterbox canvas；不声称可映射回原图 |
| R2 tag 版本 | **前半句不成立，后果成立**。`tag_version` 已写入 `layerdiff/manifest.json`，但未传播到 `optimized/manifest.json` / run fingerprint；官方 v0.0.2 配置已核实为 `v3` | 把 LayerDiff manifest 升为必需输入；registry 按版本解析；冻结 v3 候选 tag 宇宙 |
| R3 两个 `info.json` | **成立**。根文件是 Marigold 内部 schema，`optimized/info.json` 才含最终几何 | 禁止读取根文件；严格验证 `frame_size` 和每个 part 的几何字段 |
| R4 根目录 PNG | **结论成立，旧理由不成立**。根 PNG 在两种保存模式下都存在，但属于后处理前的 tag 宇宙 | 根 PNG 永远不是合法 `PartSource`；只读 final PSD 或 `optimized/` PNG |
| R5 `frame_size` 顺序 | **成立且已实测**。`PSDImage.new(size=...)` 接收 `(width, height)`；上游把 NumPy shape 前两项原样传入，方形画布掩盖了顺序问题 | v1 只接受正方形并与所有 canvas 交叉校验；非方形 fail fast，不猜字段顺序 |
| R6 C 阶段过重 | **成立，但不能省略 atlas 文件**。Spine atlas 可有多个 page | Revision 3 当时采用一部件一 page/region；该临时决定已被 Revision 5/6 的共享 plan 契约取代 |
| R7 pose 候选 | **更新**。SDPose-OOD 的 OOD/艺术域证据更直接，但集成成本显著高于 RTMW | SDPose-OOD Body 为首选评测候选；RTMW-l 为轻量 ONNX 备选；两者默认关闭 |

### Revision 5 Claude 审查核对

| 审查项 | 核对结论 | Revision 5 处理 |
|---|---|---|
| F-1 双格式成为全局阻塞 | **风险成立，改默认 Spine-only 的建议不采纳**。用户已明确正式交付必须同时含两种格式 | A-E 是内部 stage gate，不是多个可发布产品；正式 release 的关键路径在 Revision 13 固定为 `A→B→C→(D,E)→G`，E 未通过就不能正式交付 |
| F-2 LBS→keyform 无误差/体积上界 | **成立，但 Revision 5 的适用面过宽** | Revision 5 先补误差/体积上限，Revision 7 收窄到非刚性位置；Revision 8 又因无 Glue 零缝矛盾把 joint blend 移出正式 v1，只保留 E0-S 测量 |
| F-3 默认要求 blink/talk 自相矛盾 | **成立** | 默认改为 `dual_runtime_core_v1`，只要求不依赖新表情纹理的结构/骨骼动作；blink/talk/组合表情按 capability 可选，严格 avatar profile 改为 opt-in；冻结分层 blink 算法 |
| F-4 六个 joint 缺显式来源 | **成立** | 在 A 阶段为 head base/top、wrist、ankle、hand tip、toe 增加 observation/eligibility 契约，exporter 禁止补算 |
| F-5 wrist/ankle 可能系统性 unresolved | **成立** | joint 指标按可观测性分桶；geometry-only wrist 不设 recall 门槛，约束 false-resolve；ankle 只在 leg/foot 接触可见样本上计算 conditional recall |
| F-6 Live2D 一部件一纹理页不可交付 | **性能/资源风险成立，“格式一定不支持”未证实**。官方 Web sample 按 `getTextureCount()` 动态遍历，没有找到 2-4 页的格式硬上限 | 不再赌运行时容忍度：C 阶段新增共享 deterministic texture packer，正式默认固定最多 4 张 2048² 页；超限失败 |
| F-7 PSD bbox 可能被 alpha 裁小 | **当前版本未复现**。`psd-tools 1.17.4` 对带透明边的 12×10 layer 保存/重载后 bbox 仍精确为写入矩形；项目 `save_psd` 实测同样成立 | 保留 PSD bbox 精确相等契约并增加透明边 fixture；未来 psd-tools 行为变化由 fixture 拦截，不先放宽成子集 |

### Revision 6 新一轮审查核对

| 审查项 | 核对结论 | Revision 6 处理 |
|---|---|---|
| N1 标准参数被写成免费实现路径 | **成立**。Cubism 参数 ID 本身没有骨骼或变形行为 | 把参数命名与 binding 编译拆开；标准/custom 参数的任何可见效果都必须有 MOC keyform/opacity binding |
| N2 区域半径中位数放宽细端误差 | **成立** | 改为逐顶点 `E_i ≤ 0.05 × r_i`；`r_i` 来自最近 medial-axis sample 的 DT 半径，并明确以宽度归一化是刻意约束 |
| N3 2048 page 与自由 resolution 隐藏耦合 | **成立，但不采用 `canvas≤page/2` 这一不充分条件**。CLI 只做 `int` 解析，配置可写 2048；GUI 才限制到 1280。只看 canvas 边长既会误拒 1280 稀疏层，也不能保证 1024 的大量满幅层可装下 | v1 白名单冻结为 768/1024/1280；A 阶段用实际 padded bboxes 运行同一 packer dry-run，超分辨率或装不下立即失败；增加真实数据面积/成功率统计 |
| N4 E0 没验证 exp3 生效 | **成立** | E0-core 增加 expression 前后参数/像素断言，以及 motion 后依次应用 Add/Multiply/Overwrite 的数值与渲染验证 |
| N5 Core gate 与普通 CI 分级矛盾 | **成立** | 普通 CI 只强制 Python parser/golden；Core consistency 和实载统一归 opt-in release gate。普通 CI 缺 Core 为 skip，正式交付缺 Core 为硬失败 |
| N6 dev profile + partial 状态未定义 | **成立** | 增加 stage/export 状态枚举；组合结果固定为 `stage_validated_with_degradation`，并记录 `rigid_fallback_applied` |

### Revision 7 deformer 原语审查核对

| 审查项 | 核对结论 | Revision 7 处理 |
|---|---|---|
| M1 刚体骨骼被错误展开成 ArtMesh 顶点 keyform | **成立，是方向性错误**。官方 Cubism 4.2 文档明确说旋转 ArtMesh 会因线性插值收缩，应改用 RotationDeformer；嵌套 RotationDeformer 正是四肢关节的标准组合 | Rig bone 树映射为嵌套 RotationDeformer；弦割误差、adaptive stops 和 baked-position 预算只约束 warp、blink/talk 与 E0-S/未来 Glue 的非刚性形变。非均匀缩放/剪切不冒充 RotationDeformer 能力 |
| M2 texture region 粒度未冻结 | **成立** | v1 明确一份 canonical part payload 对应一个 atlas region；同 part 的连通分量和未来 Live2D skinning 子网格共享该 region/UV，接受透明空洞浪费，因此 A 的 part-bbox dry-run 仍有效 |
| M3 expression full-weight 断言受 fade 影响 | **成立** | E0-core 的 exp3 fixture 固定零 fade，至少推进一次 update 并确认 effective weight 为 1；另用底层 API 单测 blend 公式，禁止在 manager 第一帧盲断言 |
| M4 texture 面积率分母歧义 | **成立，但不只保留一个数** | 删除模糊字段，分别记录四页预算占用率 `budget_occupancy` 与已用页装填率 `used_page_fill` |
| 768/1024/1280 白名单完整性 | **已核实**。`see_through_profile.py` 的四个硬件 profile 只产生 1280、1024、768 三个唯一值 | 白名单不变，并增加 profile 集合等值测试防止上游以后新增档位而 auto-rig 漏接 |

### Revision 8 skinning 与坐标空间审查核对

| 审查项 | 核对结论 | Revision 8 处理 |
|---|---|---|
| P1 无 Glue blend band 的零缝门与插值误差门互斥 | **成立，是数学矛盾**。刚性边界走旋转圆弧，baked 边界在 stops 间走弦；只要 `ρ>0` 且 `Δθ≠0`，两者中间态就不能逐点重合。region extrude 不会修复 region 内部的几何缝 | `Live2DSkinningPartition` 移为实验性 `E0-S`，不进入正式 Live2D v1 compiler；Revision 8 暂按交集省略 `wave.*`，Revision 9 改为 Spine 保留、Live2D omitted 的显式 optional 分叉。未来把关节弯曲设为 Live2D required 前必须验证 Glue writer |
| P2 嵌套 deformer 缺逐层局部空间 | **实质成立；“评审基准没列 WARP_DEFORMERS.md”不成立，Revision 7 已列入**。该文档对 CMO3 Editor 的逆向证据支持 warp-local `0..1`，但现有 StretchyStudio MOC3 writer 只发射 root warp 且把位置统一 PPU-normalize，不能证明嵌套 MOC3 runtime 的确切单位。`scales` 本身无量纲，变化的是它在父空间中的组合效果 | 删除“单一坐标变换”，新增版本化 `Live2DCoordinatePlan`；E0-core 用 `root→warp→rotation→rotation→ArtMesh` 实载冻结 root/warp/rotation frame 的 forward/inverse 规则，证据未闭环前不得猜单位 |
| P3 同一 RotationDeformer 同时插值 angle 与 origin 未覆盖 | **成立**。只测两个嵌套参数不能证明单节点的 pivot translation 与 rotation 顺序 | 冻结 parent-local similarity transform 的分解顺序，并在 E0-core 对同一参数同时改变 angle/origin，在区间内 9 点比较 runtime 与解析变换 |
| P4 non-rigid/MOC3 预算在默认路径接近空载 | **结论部分成立**。required `breath` 仍有少量 WarpDeformer control-point keyforms，所以 baked 计数并非严格为零；但 100 万 positions 与 64 MiB 在 core profile 下通常只承担资源熔断，不衡量质量 | 保留硬上限，报告中标为 `capacity_guard`；不得纳入质量评分。质量看坐标 round-trip、runtime parity、插值误差和可见 landmark，纹理资源仍由 TexturePagePlan 单独约束 |

### Revision 9 preset parity、活性与默认姿态审查核对

| 审查项 | 核对结论 | Revision 9 处理 |
|---|---|---|
| Q1 `wave.*` 的 optional 能力分叉契约矛盾 | **成立，是策略未冻结而非 Spine bug**。Revision 8 的意图是交集，但“Spine dev 可生成”和 required-only parity 验收允许两种实现。为保留 Spine 原生能力，不继续假装 optional 必须对称 | 正式 dual profile 固定 `optional_preset_parity="per_format"`；required preset 仍必须双格式一致，optional preset 可只在 Spine 出现。`motion_manifest.json` 逐 preset/format 记录 `supported/omitted`、原因与文件引用 |
| Q2 无 driver 的 limb RotationDeformer 成为死枝 | **成立；但不接受手写 `root/torso/...` 白名单**。实际活节点由 profile、per-format preset、channel 和 attachment ancestry 决定，`upper_arm/thigh` 是否保留取决于 preset 是否真的驱动它们 | Live2D compiler 先从公共 channel 生成 `Live2DBindingPlan`，再构建 `Live2DDriverLiveness` 并只发射可达 deformer；静态 pass-through bone 的 rest transform 烘入最近 live parent/root，完整 emitted/pruned/reparent 表写报告，任何 emitted dead deformer 失败 |
| Q3 default parameter 未验证 rest pose | **成立**。结构 consistency 不保证默认参数插值结果等于作者定义的 rest mesh；但不必无条件添加第三个 key，只需证明 default-rest invariant | E0-core 增加全部参数置 default 的顶点/opacity/draw-order/rest-transform parity；两端 key插值不能还原 rest 时强制写显式 default keyform，否则报 `live2d_default_pose_mismatch` |
| Q4 coordinate schema 未签署的触发时机 | **成立**。这是 compiler/runtime 组合的启动前置，不应在每个 item 中重新发现 | E0-core 生成 digest-pinned `attestations/live2d-frames-v1.json`；job 在枚举 item 前校验并在缺失/不匹配时返回 `live2d_coordinate_schema_unverified`。Revision 10 进一步把参与 gate 的字段缩到 frame contract，item 内只报告 round-trip failure |

### Revision 10 deformer 身份、attestation 边界与符号表审查核对

| 审查项 | 核对结论 | Revision 10 处理 |
|---|---|---|
| S1 RotationDeformer 被同时写成 bone 级与 `(bone, parameter)` 级 | **成立，是输出语义不确定性**。官方 Editor 允许对象映射多个普通参数，但本设计已拒绝多参数全量 keyform grid；因此 v1 必须采用一参数一 rotation node，而不是依赖未冻结的多参数绑定行为 | 实例键固定为 `(bone_id, parameter_id)`；每个 rigid binding 必须带 `stack_rank`，同 bone 按 `rotation-stack-v1` 从小到大外→内嵌套，rank 冲突直接失败，不用字典序掩盖缺配置。E0 增加顺序反转会产生可见差异的非交换 fixture |
| S2 attestation 比 frame 契约更宽 | **主结论成立；“普通 CI 必须重签、否则不能运行”不成立**。CI 应消费已发布 attestation 并跑纯结构/向量验证，不应拥有 Core 或生成新证明；真正的问题是完整 compiler/writer hash 把无关改动误判为坐标变化 | gate 改为窄的 `live2d_frame_contract_digest`，只覆盖专用 frame/layout semantic kernels、descriptor、codec/transform/stack-order 版本、E0 vectors 和已实测 Core allowlist；完整源码摘要只作 provenance/cache，不参与 coordinate gate。普通 CI 可跑 structural tier，但不能写正式 completed manifest |
| S3 Live2D 缺少与 Spine 等价的稳定 ID 契约 | **成立**。参考项目逆向表明确列出 Parts/Deformers/ArtMeshes/Parameters 的 `ids` section；即使字段名将来修订，跨文件引用仍要求稳定 ID | `GlobalExportSymbolTable` 在任何格式剪枝前从完整 symbol universe 生成一次；Spine 与 Live2D 共用映射。MOC3 四类 ID、model3/cdi3、motion/expression 和 artifact 引用全部闭环，缺映射或碰撞统一失败 |

### Revision 11 rank 分层与 candidate-superset 审查核对

| 审查项 | 核对结论 | Revision 11 处理 |
|---|---|---|
| T1 rank 重复依赖 torso/head 永不共用 bone | **成立，是默认 profile 的演进陷阱**。当前 preset 恰好分骨并不能成为 registry 不变量；未来 idle 增加 head channel 就会让两个 required parameter 在同 bone 撞 rank | built-in rigid-driver registry 的 rank 改为全局唯一 `100/200/300/400`，并在 registry load/CI 直接断言全局唯一。rank 仍只决定 same-bone stack，但跨语义域复用数值被禁止 |
| T2 rank 表内容进入 frame attestation | **成立，Revision 10 又把内容指纹塞回了 E0 gate**。E0 证明 runtime 按 parent chain 组合的数学语义；新增 production driver/rank 只改变编译输入，不改变 Cubism frame/runtime 模型 | `live2d_frame_contract_digest` 只覆盖 `lower rank=outer`、冲突规则、parent chaining 和 synthetic non-commuting E0 fixture；`RigidDriverRegistry` 行内容/摘要只进入 motion/export fingerprint。改表必须重编译和跑 structural tests，但不重签 E0 |
| T3 candidate universe 超集只靠生产时失败 | **成立**。`missing_export_symbol` 应是最后防线，不应是首次发现两路径漂移的主要机制 | E 不再构造 primitive key，只能筛选 C 同一 enumerator 产生的 immutable candidate records；再加 generated Rig × 全 profile × 全 preset 属性测试，断言 `binding_plan_keys ⊆ candidate_universe` |

### Revision 12 stage ownership、命名空间与启动错误审查核对

| 审查项 | 核对结论 | Revision 12 处理 |
|---|---|---|
| U1 B/C 共同改写 `rig.json` 使 resume 永久失效 | **成立，是 artifact ownership 缺失导致的正确性问题**。完整 `RigDocument` 含 C 字段，B 不可能既拥有它又让自己的 output digest 在 C 后保持不变 | 采用单写者方案：A/B 只写各自私有 cache；B 的正式输出是 `RigGeometryCache v1`，不是残缺 `RigDocument`；C 在成功末尾一次性组装、验证并原子写入完整 `rig.json`。每个路径恰好一个 owner stage，任何 stage 不得把下游会改写的路径列为 output |
| U2 整个 symbol universe 共用一个唯一名称池 | **主结论成立，但 `(format, namespace)` 仍不足以表达 Spine attachment 的局部作用域**。Spine attachment map key 由 skin/slot/name 组成，而 actual attachment name 与 atlas region 又是不同引用域；MOC3 四类 ID 也不是一个扁平数组 | 引入 canonical typed `ExportNamespaceKey`；唯一性检查按完整 namespace key 分组，attachment-key namespace 包含 skin/slot scope。typed key 到 `{namespace_key, export_name}` 的函数及跨格式 `base_export_name` 仍全局稳定；跨 namespace 同名合法且不加无意义后缀 |
| U3 built-in registry load 错误被写成 item diagnostic | **成立**。它发生在枚举 item 之前，没有 part/joint，也不应受 `continue_on_error` 控制 | built-in registry 重复 rank 是 CI invariant；worker 仍 defense-in-depth 地在启动期校验，失败用 job-level `invalid_rigid_driver_registry` 非零退出且不创建 item staging/report。`live2d_deformer_order_conflict` 只保留给已有 item 内 binding plan 的实例/rank 自相矛盾 |

### Revision 13 终态、失败产物与 canonical PNG 审查核对

| 审查项 | 核对结论 | Revision 13 处理 |
|---|---|---|
| V1 `export_manifest.json` 的 owner `join` 不是 stage | **成立，且会让 item-level `skip_completed` 绕过 stage DAG**。仅凭 completed 文件存在跳过，会在 D/E fingerprint 已变化时继续交付旧包 | 新增真正的 G terminal-finalization stage 与 `rig/cache/G/manifest.json`。G manifest 绑定当前 C/D/E manifest、artifact-set、profile 和 validator 摘要；`skip_completed` 必须递归验证 A-E/G manifest 与输出，不能只看 `export_manifest.json`。旧 G 无效时先撤销终态文件，再从最早失效 stage resume |
| V2 A/B 失败没有公共 per-item 诊断 | **成立；但不采用“实际失败 stage 动态拥有同一路径”**，那会重新引入跨运行多写者 | G 是 `rig/error.json` 的唯一 writer，runner 对任何已枚举 item 的 A-E 失败都调用 G failure finalization。`error.json` 与 `export_manifest.json` 互斥；前者公开、可寻址、cache 删除后仍保留，后者只在完整成功时存在 |
| V3 D/E 分别编码却要求 PNG 字节完全相同 | **成立，当前只有断言没有 construction path** | C 在 `rig/shared/textures/` 用固定 encoder 一次性写 canonical PNG，并记录 raw RGBA 与 encoded-file 两个 SHA；D/E 禁止 decode/re-encode，只做 byte copy，目标 SHA 必须等于 C。encoder 版本/设置进入 C fingerprint |

### 决策摘要

新增 `module/auto_rig/`，消费 see-through item 目录，生成版本化 `RigDocument`，并从同一份
Rig 批量导出带预设动作/表情的 Spine 4.2 包和 Live2D Cubism runtime 包。它不是编辑器、
不是预览工具，也不是 see-through 内部第四阶段；不生成 GIF/WebM，也不把交互式预览作为
产品功能。

实现分七个具名 stage gate；它们可独立测试和缓存，但不是七个可独立发布的产品：

1. **A：输入适配 + 掩膜几何 + 关节质量报告 + overrides**；
2. **B：网格 + 通用多骨骼权重 + 私有 `RigGeometryCache v1`**；
3. **C：能力驱动的 exporter-neutral 动作/表情 preset 绑定 + Rig 级全局符号表 + 共享纹理页规划/canonical PNG，并组装/写入完整 `RigDocument v1`**；
4. **D：Spine 4.2 setup rig、animations、简单多页 atlas 导出与验证**；
5. **E：Live2D MOC3 V4.00 runtime、motions、expressions、纹理导出与验证**；
6. **F：可选姿态后端评测，SDPose-OOD Body 为首选候选、RTMW-l 为备选，不阻塞 A-E/G**；
7. **G：item terminal finalization；成功时验证并发布双格式 `export_manifest.json`，失败时发布公共 `error.json`**。

姿态模型默认关闭。没有几何基线和人工标注评测集之前，不把“SDPose-OOD 替换 DWPose”
写成既成事实。正式批处理 profile 固定要求 Spine 4.2 与 Live2D runtime **同时成功**；单格式
开关只用于开发诊断，不能把只成功一半的 item 标为 completed。正式 release 的依赖图是
`A → B → C → (D, E) → G`；任一 A-E failure 也走 failure edge 到 G 写终态错误。D/E 可并行验收，
但 E 未完成时项目就尚未达到正式交付定义。F 是实验分支，不在 release 关键路径。

实施顺序不按字母机械推进。由于 E 是最大未知项，开发大规模 A-D 之前先做 **E0 可行性门**。
正式路径的最小 **E0-core** MOC3 V4.00 fixture 必须同时包含：

1. 一个 root ArtMesh、一个由 `ParamBreath` 驱动的 root WarpDeformer、两个由不同参数驱动且位于
   warp 下的父子 RotationDeformer，以及分别直接挂在 warp 和最内层 rotation 下的 ArtMesh；在
   rest、单参数和三个参数同时非默认值时验证 parent indices、每层 local frame 与最终顶点；
2. 一个单独的 RotationDeformer，其同一参数 keyforms 同时改变 `angle` 与 `origin_x/y`；以规范的
   parent-local similarity transform 顺序在区间内 9 点验证，不能拿嵌套参数测试代替；
3. 同一 Rig bone 上两个由不同 parameter 驱动的 RotationDeformer，使用不同 pivot/origin，且两者
   同时非默认值时正序与逆序的解析顶点差必须 `>0.1 px`；fixture 使用 synthetic ranks `10/20`，
   runtime 只能匹配 `rotation-stack-v1` 的 lower→outer 顺序，parent indices、instance IDs 和报告顺序
   也必须一致。fixture 不导入 production `RigidDriverRegistry` rows，不能只断言输出字节稳定；
4. 一个 default 严格位于 min/max 内部且不在中点的参数、一条完全无 driver 的静态 limb bone/mesh
   分支，以及一条 `live parent → static intermediate → driven child` 链。正例在需要时显式写 default
   keyform、删除完整死枝，并把静态中间节点的 rest frame 正确折叠进 driven child；负例故意缺
   default-rest binding，必须在模型初次 `csmUpdateModel()` 后被检测为非 rest；
5. 一条 `.motion3.json`、分别覆盖 `Add/Multiply/Overwrite` 的三条 `.exp3.json` 和一张纹理页。

另设 **E0-S** 实验 fixture：两骨肢体、两个刚性 ArtMesh 与一个无 Glue baked blend band。它必须
量出 stop 间的弦割偏差和 render-space seam，而不是要求数学上不可能的逐点零缝；结果只决定
`experimental_live2d_joint_bend` 是否继续研究，不属于 E0-core、正式 release 或 item 完成条件。

普通 CI 先强制通过 Python parser/golden；配置了官方 Core/SDK 的 E0-core release gate 再通过
`csmHasMocConsistency`、非空渲染、逐层坐标 round-trip、嵌套 deformer 组合和 runtime transform
parity。加载后、任何 motion/expression/physics 应用前，把全部 parameter 显式设为 MOC3 default，
执行一次 `csmUpdateModel()`；所有 drawable vertices、opacity、draw order 和解析 deformer rest
transform 必须与编译前 rest 数据相差 `≤0.1 px` 或对应标量容差。expression 公式
分两层验证：底层 CubismModel API 单测在 full weight 下断言 `Add: p+v`、`Multiply: p*v`、
`Overwrite: v`；端到端 exp3 fixture 固定 `FadeInTime=0`、`FadeOutTime=0`，至少推进一次 runtime
update，并在确认 effective expression weight 已为 `1` 后才做参数值与 landmark/像素断言，禁止
在 manager 启动第一帧直接比较。更新顺序固定为 runtime 示例的
`load → motion → save → expression → model update/render`。E0-core 同时记录可复现的 SDK/Core 获取与
许可证方案；release gate 未执行或失败时，正式双格式项目就是 blocked，A-D 即使继续也只能算
内部研发产物，不能把 Spine-only 改名成交付完成。

### 输入适配与冻结契约

auto-rig 的所有几何坐标都处于 **LayerDiff canvas 空间**。`layerdiff_core.py:179-181` 先调用
`center_square_pad_resize`，再把结果保存为 `src_img.png`；因此当前 canvas 恒为
`resolution×resolution` 方形 letterbox 图，默认 `1024×1024`，高档 profile 为
`1280×1280`。`fullpage_pad_size`、`fullpage_pad_pos` 和缩放因子没有落盘；虽然
`layerdiff/manifest.json` 保留了外部 `source_path`，它不足以构成自包含、可复现的反变换契约。
Rig v1 不提供原图坐标，导出的纹理也来自同一 canvas 闭环。以后确需原图坐标时，应由
see-through 新增版本化 `source_to_canvas` 仿射矩阵，而不是 auto-rig 根据文件尺寸猜。

共同必需文件：

| 文件 | 契约 |
|---|---|
| `src_img.png` | 方形 letterbox canvas 和姿态先验输入；不得从 amodal 图层重合成 |
| `layerdiff/manifest.json` | `tag_version`、`resolution` 和 LayerDiff 原始 part 清单；缺失或未知版本即失败 |
| `optimized/info.json` | 后处理最终 tag、`xyxy`、`depth_median`、`frame_size` 的唯一语义事实源 |
| `optimized/manifest.json` | 判定 `save_to_psd`、`tblr_split` 与产物布局 |

合法 payload 二选一：

| 模式 | 部件 RGBA | 部件深度 | 说明 |
|---|---|---|---|
| PSD，当前默认 | `final.psd` 图层 | `final_depth.psd` 同名图层 | 用 `info.json` tag 匹配图层；PSD 名称只作 lookup，不重新解释语义 |
| PNG | `optimized/<tag>.png` | `optimized/<tag>_depth.png` | `save_to_psd=false` 时的布局 |

根目录的 `info.json` 是 Marigold 内部文件，`parts[tag]` 可能只是空对象；根目录的
`<tag>.png` / `<tag>_depth.png` 在两种保存模式下都会存在，但属于 `further_extr` 之前的 tag
集合。它们即使尺寸和文件名看似合理，也**永远不是合法 `PartSource`**。例如后处理把
`handwear` 替换成 `handwear-r/-l` 后，根目录仍可能只有未拆分的 `handwear.png`。读取这些
文件会得到“有 tag、无最终几何”或纹理与 tag 不对应的静默错误，必须以明确错误拒绝。

当前 v0.0.2 模型的官方 UNet 配置为 `tag_version="v3"`。v3 的候选 tag 流转冻结如下；
“候选”不表示每张图都存在，近空 mask 会被 `load_part` 丢弃：

| 阶段 | v3 tag 契约 |
|---|---|
| LayerDiff 原始输出（24） | `front hair`, `back hair`, `head`, `neck`, `neckwear`, `topwear`, `handwear`, `bottomwear`, `legwear`, `footwear`, `tail`, `wings`, `objects`, `headwear`, `face`, `irides`, `eyebrow`, `eyewhite`, `eyelash`, `eyewear`, `ears`, `earwear`, `nose`, `mouth` |
| Marigold/后处理基础候选（23） | 上述集合去掉 `head`；`head` 不在 `VALID_BODY_PARTS_V2`，不会写深度或进入 `parts` |
| `tblr_split=false` 最终集合 | 23 个基础候选的非空子集 |
| `tblr_split=true` 最终集合 | 对 `handwear`, `eyewhite`, `irides`, `eyelash`, `eyebrow`, `ears` 各自保留原 tag，或在检测到至少两个前景连通域时用面积最大两块替换为 `-r/-l`；不是无条件追加 12 个 tag，可靠性需 auto-rig 复验 |

v3 中没有 `hair`、`eyes`、`hairf` 或 `hairb`。`front hair` / `back hair` 是正式 canonical
tag。`CanonicalTagRegistry` 必须由 `tag_version` 选择，`BoneSpec.requires` 只引用 registry
定义的 canonical ID，不能写一个当前版本永远不会满足的别名。首版只接受 v3；未来支持 v2
必须新增独立 fixture 和显式 registry 分支，不能把两个版本的 tag 做模糊 union。

`module/auto_rig/contracts.py` 读取时冻结这些细节：

- `optimized/info.json` 顶层必须有对象 `parts` 和长度为 2 的数值数组 `frame_size`；
- 每个 `parts[tag]` 必须是对象并包含 `tag`、四元素 `xyxy` 和有限数 `depth_median`；key 与
  内部 `tag` 必须相同，字段缺失不能退化为空几何；
- `frame_size` 来自 NumPy `shape[:2]`，但上游 PSD writer 又把两项传给要求 `(width,height)`
  的 `PSDImage.new`。v1 先断言两项相等，再建立 `{width: n, height: n}`；非方形输入报
  `unsupported_non_square_frame`，不宣称已验证一般情况下的字段顺序；
- `xyxy = [x1, y1, x2, y2]`，右下角为 NumPy slicing 的 exclusive 边界；
- canvas 坐标原点左上，`+x` 向右，`+y` 向下，单位为 `src_img.png` 像素；
- `depth_median` 越小越靠前；读取后统一转 Python `float`；
- `layerdiff/manifest.json.resolution`、`frame_size`、`src_img.png`、两份 PSD canvas 必须一致；
- auto-rig v1 只接受 square canvas edge `{768, 1024, 1280}`。See-through CLI 虽然接受任意
  `int`，但 2048 等自定义值不属于本 exporter profile；A 阶段立即报
  `unsupported_auto_rig_canvas_resolution`，不能等 C 阶段装箱才失败；
- 最终 payload tag 集合必须与 `optimized/info.json.parts` 精确相等；PNG 尺寸必须等于自身
  `xyxy` 的宽高，PSD 图层位置/尺寸必须还原到该 `xyxy`；任何不一致直接报错，不猜。

fixture 至少覆盖 v3 PSD、v3 PNG、左右成功拆分、左右未拆分、误传根 `info.json`、误传根
PNG、768/1024/1280 合法 canvas、2048 被拒绝和伪造非方形 `frame_size`。这组 fixture 同时把
`optimized/info.json` 从内部产物升级为可消费契约；调用方不得绕过 parser 读取裸 JSON。

参数顺序的证据缺口已补：2026-07-31 用 `psd-tools 1.17.4` 实测
`PSDImage.new(mode="RGBA", size=(37, 19))` 得到 `width=37, height=19`。方形生产画布下没有
行为错误，但非方形支持在修正/迁移上游 writer 之前不能开放。

PSD bbox 的疑点也已实测而非猜测：用 12×10 RGBA layer、仅中央 5×4 区域非透明，按
`left=7, top=6` 写入并保存/重载，`psd-tools 1.17.4` 返回的 layer bbox 仍为
`(7, 6, 19, 16)`，没有按 alpha 裁成更小矩形；直接调用本仓库 `save_psd` 得到相同结果。
因此 v1 继续要求 PSD stored layer rectangle 与 `xyxy` 精确相等，并用带透明边 fixture 防止
依赖升级后行为漂移。alpha-tight 内容 bbox 可以是其子集，但不能拿它替代 layer rectangle。

分辨率是 A 阶段的硬验收风险，不是以后再优化：人物在宽幅/竖幅原图 letterbox 后可能只占
canvas 中间窄带，默认 1024 下手指、耳朵、发梢和窄肢体会接近像素极限。几何算法必须先用
身体 mask union bbox 工作，低于相对宽度阈值的细部只允许刚性绑定或 `unresolved`。A 阶段
要在 768/1024/1280、极端长宽比和小主体样本上分别报告 joint/contour 可用率；过不了就不能以
“提高模型精度”掩盖输入已经丢失的空间信息。

### 唯一事实源与产物

`rig/rig.json` 是唯一公共事实源，且 **C 是它的唯一 writer**。A/B 只在 `rig/cache/` 写带独立
schema 的私有阶段产物；它们可以删除重算，exporter 不得直接读取。每个文件路径必须恰好属于一个
owner stage，并且只能出现在该 stage manifest 的 `output_file_sha256[]` 中；下游只能把它的摘要记录为
input/upstream dependency，不能改写文件后再把同一路径声明成自己的 output。唯一的结构性例外是
`rig/cache/<stage>/manifest.json` 本身：它是该 stage 最后写入的 commit marker，不把自己的 SHA 放进
自身 output 列表以避免循环摘要；其文件 SHA 由直接下游记录，terminal G 则由
`StageGraphValidator` 直接验证。

| 文件 | owner | 契约 |
|---|---|---|
| `rig/cache/A/geometry_observations.json` | A | 私有 `GeometryObservationCache v1`；parts、joint observations、dry-run 与诊断证据 |
| `rig/cache/B/rig_geometry.json` | B | 私有 `RigGeometryCache v1`；canvas/parts/joints/bones/meshes/weights 与 A/B diagnostics，不是 `RigDocument` |
| `rig/cache/G/manifest.json` | G | terminal input/output digest、成功/失败状态与当前 C/D/E manifest/artifact-set 摘要；G commit marker |
| `rig/cache/<stage>/manifest.json`、`rig/cache/<stage>/*` | 对应 stage | 私有缓存指纹、产物摘要、中间数组和调试图；路径不得跨 stage 共写 |
| `rig/rig.json` | C | 完整、公共、版本化 `RigDocument v1`；C 成功前不存在，D/E 只读 |
| `rig/report.json` | C | A-C 结构化 diagnostics、质量指标和全局符号表摘要；D/E 不追加，映射本体在 `rig.json` |
| `rig/motion_manifest.json` | C | 公共 preset 定义、逐格式 supported/omitted 决策、原因、文件引用和默认 clip 建议 |
| `rig/shared/textures/page_<index>.png` | C | canonical encoded PNG bytes；D/E 只允许 byte copy，不允许解码重编码 |
| `rig/export_manifest.json` | G | 成功终态；C/D/E manifest、G finalizer version、profile/validator 与全部发布 artifact-set 摘要 |
| `rig/error.json` | G | 失败终态；ordered failed stages/records、稳定 diagnostics、input/config/override fingerprint 与可执行修复建议 |
| `rig/spine/skeleton.json` | D | Spine 4.2 数据 |
| `rig/spine/skeleton.atlas` | D | Spine atlas |
| `rig/spine/textures/page_<index>.png` | D | 从 C canonical page 原样复制的 Spine atlas pages；SHA 必须相同 |
| `rig/spine/export_report.json` | D | global symbol table 的 Spine 子集/摘要、validator 结果和降级记录 |
| `rig/live2d/<model>.moc3` | E | MOC3 V4.00 二进制模型 |
| `rig/live2d/<model>.model3.json` | E | runtime 入口与资源引用 |
| `rig/live2d/<model>.cdi3.json` | E | 参数、部件的显示信息 |
| `rig/live2d/textures/page_<index>.png` | E | 从 C canonical page 原样复制的 Live2D 纹理页；SHA 必须相同 |
| `rig/live2d/motions/<preset>.motion3.json` | E | 固定动作 preset |
| `rig/live2d/expressions/<preset>.exp3.json` | E | 固定表情 preset |
| `rig/live2d/export_report.json` | E | global symbol table 的 Live2D 子集/摘要、参数映射、MOC consistency、runtime 验证和降级记录 |

正式完成标记至少包含：

```json
{
  "schema_version": 1,
  "producer_stage": "G",
  "finalizer_version": "terminal-finalizer-v1",
  "input_fingerprint": "sha256:...",
  "profile": "dual_runtime_core_v1",
  "profile_fingerprint": "sha256:...",
  "required_formats": ["spine_4_2", "live2d_moc3_v4_00"],
  "upstream_stage_manifests": {
    "C": "sha256:...",
    "D": "sha256:...",
    "E": "sha256:..."
  },
  "upstream_artifact_sets": {
    "canonical_textures": "sha256:...",
    "spine_4_2": "sha256:...",
    "live2d_moc3_v4_00": "sha256:..."
  },
  "motion_manifest_sha256": "sha256:...",
  "global_symbol_table_sha256": "sha256:...",
  "validation": {
    "tier": "release",
    "spine_validator_fingerprint": "sha256:...",
    "live2d_validator_fingerprint": "sha256:..."
  },
  "formats": {
    "spine_4_2": {"status": "validated", "files": []},
    "live2d_moc3_v4_00": {"status": "validated", "files": []}
  },
  "status": "completed"
}
```

`formats.*.files[]` 的每项至少是 `{path, size, sha256}`，按 path 排序后形成对应 artifact-set digest；
不能只记录目录名或文件数量。G manifest 再把上述 `export_manifest.json` 的最终 SHA 作为自己的 success
output，因此不在 export manifest 内反向记录 G manifest SHA，避免循环摘要。

公共失败终态至少包含：

```json
{
  "schema_version": 1,
  "producer_stage": "G",
  "terminal_state": "failed",
  "item_id": "item/...",
  "input_fingerprint": "sha256:...",
  "config_fingerprint": "sha256:...",
  "rig_overrides_sha256": "sha256:...",
  "failed_stages": ["D", "E"],
  "failure_records": [
    {"stage": "D", "sha256": "sha256:..."},
    {"stage": "E", "sha256": "sha256:..."}
  ],
  "failure_set_sha256": "sha256:...",
  "diagnostics": [],
  "retryable": true
}
```

各阶段先写 stage-local staging 目录再替换自己的产物；禁止 B 写一个半成品 `rig.json` 再由 C 原地补
字段。C 从已验证的 A/B cache 与 C 自己的 binding/packing 结果组装完整文档，在 staging 中通过
`RigDocumentValidator` 后才逐文件原子替换 `rig.json`、`report.json`、`motion_manifest.json` 与 canonical
texture pages，并把 C manifest 作为最后一个 commit marker 写入。D/E 只有在 C manifest 与全部 C-owned
文件摘要匹配后才能
启动；中途崩溃只会使 C 失效重跑，不能暴露一个可复用的 partial Rig。

G 是每个**已枚举 item** 的终态 stage，由 runner 的统一 `finally` 路径调用；D/E fan-out 必须先全部
settle（成功、失败或明确取消）并收集 failure records，同一 item 的 G 恰好执行一次：

1. success path 重新验证当前 C/D/E manifest 的 expected fingerprint、所有 output SHA、required
   capability、profile 与 validator 结果，构造完整 artifact-set digest，再写
   `export_manifest.json`；
2. failure path 接受一个或多个失败 stage 的规范化 `StageFailureRecord`，按 stage name 排序、形成
   failure-set digest 后写公开、稳定、per-item 可寻址的
   `error.json`；A/B/C/D/E 不能直接写这个路径；
3. G-owned `export_manifest.json` 与 `error.json` 在一次完成的 terminal state 中严格 XOR。success
   先删除旧 error，failure 先删除旧 export manifest；目标文件逐文件原子替换，G manifest 最后写入；
   success manifest 的 `output_file_sha256[]` 只列 export manifest，failure manifest 只列 error record；
4. job-startup gate 在 item 枚举前失败时仍没有 per-item `error.json`，只返回上一节定义的 job-level
   error；F 实验评测失败也不改变正式 A-E/G 终态；
5. 导出失败时允许保留 C 公共产物和 D/E 各自报告供重试，但不得为汇总错误去改写 C-owned
   `rig/report.json`。G 从 stage-local failure record 复制稳定诊断，CLI/job result 只负责批次聚合。

`skip_completed` 不是 `Path.exists()`。它必须调用与正常 resume 相同的 `StageGraphValidator`，从当前
input/config/code/registry/profile 重新计算 A-E/G expected fingerprints，递归核对每份 stage manifest、
全部 output SHA，以及 `export_manifest.json` 内的 C/D/E manifest 和 artifact-set 摘要。只有 G manifest
状态为 completed、其 output digest 匹配且整条 DAG 可复用时才跳过 item。`export_manifest.json` 单独
存在、只有 `status=completed`、G cache 被删、D/E encoder/validator 改版或任一 artifact 被篡改，都
不得触发 skip；若只有 G cache 缺失而 A-E 全部有效，只重跑廉价 G，不重算 mesh/exports。
一旦 validator 判定 G 不可复用，调度器必须在启动最早失效 stage 前调用 G-owned
`invalidate_terminal()`，移除旧 G manifest、`export_manifest.json` 和 `error.json`；其他 stage 不能删除或
覆盖这些路径。这样运行过程中不会继续暴露一个已知陈旧的 completed 标记。

状态不是自由文本，v1 冻结为三组互不混用的枚举：

- A-F stage report：`stage_validated`、`stage_validated_with_degradation`、`stage_failed`；
- G manifest：`completed`、`completed_with_degradation`、`failed`；
- 公共 terminal artifact：成功只写带 `completed*` 的 `export_manifest.json`，失败只写
  `terminal_state=failed` 的 `error.json`。失败 stage 的私有 report 仍保留作深度诊断，但不是唯一记录。

没有刚性降级才可使用 `stage_validated` / `completed`。`spine_4_2_dev --allow-partial` 若实际采用
任何允许的刚性 fallback，固定写 `stage_validated_with_degradation` 并记录
`rigid_fallback_applied`，但仍永远不能写正式 `export_manifest.json`。正式双格式 profile 与
`--allow-partial` 叠加时，只有两种格式都通过 validator、所有 required capability 仍满足，才写
`completed_with_degradation`；刚性降级若破坏 required preset，仍以 `dual_export_incomplete`
失败。`--allow-partial` 从不把格式校验失败改写成成功。

### `RigDocument v1` 数据契约

`RigDocument v1` 只表示 **C-complete** 的公共文档，不存在“B 阶段的部分 RigDocument”。B 输出的
`RigGeometryCache v1` 使用独立 `cache_schema_version` 和 validator；它可以包含下列几何字段，但不能
伪装成 `schema_version=1` 的公共 Rig，也不能被 D/E 接受。C 是唯一负责把 A/B cache、capability、
motion/expression、candidate/symbol table 和 texture plan 组装为以下完整结构的阶段。

首版至少包含：

```json
{
  "schema_version": 1,
  "generator": {"name": "qinglong-auto-rig", "algorithm_version": "1"},
  "input_fingerprint": "sha256:...",
  "input": {"tag_version": "v3", "coordinate_space": "layerdiff_canvas"},
  "canvas": {"width": 1024, "height": 1024, "origin": "top_left", "y_axis": "down"},
  "parts": [],
  "joint_observations": [],
  "joints": [],
  "bones": [],
  "meshes": [],
  "capabilities": [],
  "clips": [],
  "expressions": [],
  "primitive_candidates": {
    "schema_version": 1,
    "enumerator_version": "primitive-candidates-v1",
    "candidate_universe_sha256": "sha256:...",
    "items": []
  },
  "export_symbols": {
    "schema_version": 1,
    "namespace_schema_version": "export-namespaces-v1",
    "exporter_families": ["spine_4_2", "live2d_moc3_v4_00"],
    "symbol_universe_sha256": "sha256:...",
    "symbols": []
  },
  "texture_pages": [],
  "diagnostics": []
}
```

硬约束：

- 内部引用只使用稳定 ID，例如 `part/handwear.xmin`、`joint/elbow.xmin`；显示名称不参与引用；
- `joint_observations` 保留 geometry、pose、override 的原始证据和各自分数；
- 最终 joint 有 `status: resolved | unresolved | overridden`、最终坐标、决策方法和证据 ID；
- bone 存 `{id, parent_id, head_joint_id, tail_joint_id, role}`，不只存一个 pivot；
- mesh rest vertex 使用 canvas 坐标；UV 使用部件纹理局部 `[0,1]`；triangle 是扁平索引；
- 每顶点 `influences` 是 `[{bone_id, weight}]`，权重非负、和为 1，首版最多 4 个；
- `xmin/xmax` 表示图像空间事实；`anatomical_side` 单独可空，不能混成一个字段；
- 未识别和 merged 部件保留在 parts/diagnostics 中，不能静默归 root 后消失；
- capability 由已解析的 part/joint/bone/mesh 事实推导，preset 只能消费 capability，不能为满足
  preset 反过来伪造骨骼；
- clip/expression 使用稳定 target ID 和 exporter-neutral channel，不直接塞 Spine timeline 或
  Live2D parameter 字段；同一个 preset 的语义与 channel 只有一个事实源，但 optional preset 的
  `supported_formats` 可以是 required formats 的真子集，必须由公共 motion manifest 明示；
- `primitive_candidates` 是 C 阶段由完整 Rig/registry 一次性生成的 immutable candidate records；
  D/E 只能按 `candidate_id` 筛选，不能重新派生 typed primitive key；
- `export_symbols` 是 C 阶段冻结的完整 `GlobalExportSymbolTable`，包含 schema/version、symbol-universe
  摘要和 typed internal key 到 canonical namespace + ASCII export name 的一对一映射；D/E 只能读取
  子集，不能新增或改变 namespace/name；
- `texture_pages` 记录 page 尺寸、part rect、padding/extrude、UV 和像素摘要；它是 C 阶段从
  parts 确定性派生的共享计划与 canonical byte payload，D/E 只能复制字节并翻译引用，不能各自重新
  装箱或编码。每页 canonical path 必须位于 `rig/shared/textures/`，同时携带
  `rgba_sha256/encoded_png_sha256/encoder_fingerprint`；指向 cache、Spine 或 Live2D 输出目录均非法。

`RigGeometryCacheValidator` 只验证 B-owned 几何/权重结构；`RigDocumentValidator` 则必须要求
`capabilities/clips/expressions/primitive_candidates/export_symbols/texture_pages` 六组 C-owned 字段存在并
满足各自 schema/digest/引用闭环，不能把空缺字段解释成“尚未运行 C”的合法状态。C 在保存
`rig.json` 后必须重新加载并通过完整 validator，防止内存模型和磁盘模型漂移；D/E 也先运行同一完整
validator，任何部分文档立即 `input_contract_mismatch`，不尝试从 cache 补齐。

#### 全局导出符号表

`GlobalExportSymbolTable v1` 在 C 阶段、任何 Spine/Live2D 剪枝之前生成。symbol universe 是以下
集合的并集，而不是某个 exporter 最终“存活节点”的集合：

1. 完整 `RigDocument` 的 part/joint/bone/mesh/clip/expression internal IDs；
2. 版本化 preset 与 parameter registry 的全部公共 IDs；
3. 按 symbol-table schema 固定的 exporter family universe
   `{spine_4_2, live2d_moc3_v4_00}`，从完整 Rig 与完整 preset registry 推导的所有 typed derived keys，
   包括 Live2D `(bone_id, parameter_id)` rotation instances、warp/ArtMesh/Part，以及 Spine
   bone/slot/attachment；不得按本次 profile、required formats、capability 或 pruning 缩小集合；
4. 纹理 page/region 和 motion/expression artifact 的稳定 key。

因此 `spine_4_2_dev` 与正式 dual profile 对同一 Rig 也必须得到同一全局表；未来新增 exporter family
需要升级 symbol-table schema，而不是悄悄改变旧名称。

C 与 D/E 不各写一套派生规则。C 调用一次 `PrimitiveCandidateEnumerator`，以“不过滤 capability”的
模式生成 immutable `PrimitiveCandidateSet`；每条 record 至少包含
`candidate_id/typed_primitive_key/binding_template/required_rig_facts/driver_registry_entry_id`。candidate
set 写入 `RigDocument.primitive_candidates`，`GlobalExportSymbolTable` 直接消费其中的 typed keys。

D/E 的 binding-plan builder API 只接受 candidate records 与 `motion_manifest`，返回所选
`candidate_id` 及模板实参；它们没有“从 bone/parameter 新建 typed key”的入口。所有 plan bindings 的
`candidate_id` 与 `primitive_target_id` 因而按构造属于 C 的 universe。D/E 若收到未知 candidate、发现
record 与 symbol table 摘要不一致，说明持久化数据或 registry/schema 漂移，必须
`missing_export_symbol` 失败，不能现场补 key/名称。运行期检查保留为 defense-in-depth，不代替 CI
的 superset 属性测试。

每个 Rig 源实体先得到与格式无关的 `base_export_name`。如果 Spine/Live2D 都对该实体做一对一表示，
两边必须使用这个相同名称；一对多派生实体则在同一 base 上追加由 typed derivation key 决定的稳定
suffix。例如 Spine 的 `bone/torso` 使用 torso base，而 Live2D
`(bone/torso, ParamAutoIdle)` rotation instance 使用同一 base 加 parameter suffix，不能因 Live2D 剪枝
后占用集合变小而换一个 base。

internal/derived key 不是靠拼接未转义字符串猜类型，而是 canonical typed record，例如
`{kind:"rotation_deformer", source_internal_ids:["bone/torso"], parameter_id:"parameter/ParamAutoIdle"}`。
每条 symbol 记录 `kind/source_internal_ids/namespace_key/base_export_name/export_name`。
`namespace_key` 同样是 canonical typed record，不能由点号或斜杠字符串拆分恢复语义。
其中 `skin_id/slot_id/directory` 等 scope 字段使用冻结的 internal ID 或 schema enum，不能使用尚待
消解的 export name，否则 namespace 分组和名称生成会形成循环依赖。

唯一性键冻结为 `(canonical ExportNamespaceKey, export_name)`，不是整个 universe 的裸
`export_name`。这里的 “Global” 只表示所有格式在 C 阶段一次生成、共享同一 typed universe 和 digest，
不表示所有格式/section 共用一个字符串命名空间。v1 至少区分：

| `ExportNamespaceKey` | 唯一性作用域 |
|---|---|
| `{format:"spine_4_2", namespace:"bone"}`、`slot`、`skin`、`animation` | 各自在 skeleton 内全局唯一 |
| `{format:"spine_4_2", namespace:"attachment_key", skin_id, slot_id}` | 同一 skin + slot 的 attachment map key；不同 slot 可合法同名 |
| `{format:"spine_4_2", namespace:"attachment_object"}` | 显式 actual attachment `name`；按 Spine JSON 契约在 skeleton 内唯一 |
| `{format:"spine_4_2", namespace:"atlas_region"}` | 整份 atlas 的 region lookup key |
| `{format:"live2d_moc3_v4_00", namespace:"part"}`、`"deformer"`、`"artmesh"`、`"parameter"` | MOC3 对应 ID section 分别唯一，不跨 section 制造后缀 |
| `{format, namespace:"motion", directory}`、`"expression"`、`"texture_page"` | 对应 manifest section/输出目录内唯一；完整相对路径另做碰撞检查 |

输出名统一为非空 ASCII、UTF-8 长度 `<64` bytes。Live2D standard parameter IDs 只在 Live2D
`parameter` namespace 中保留；普通项只在同一 namespace 碰撞时附加稳定 hash suffix。因而同一
`topwear` 源实体派生的 Spine slot、attachment key、atlas region、Live2D Part 和 ArtMesh 可以都叫
`topwear`，而不会互相消耗名称；同一 namespace 内两个不同 typed key 仍必须消解或失败。

相同 typed key 在任何 profile、剪枝结果、MOC3/model3/cdi3、Spine JSON/atlas、motion/expression 和
报告中必须解析成同一个 `{namespace_key, export_name}`；共享 `source_internal_id` 的跨格式实体还必须
能通过 `base_export_name` 交叉对照。Spine exporter 必须计算
`effective_region_path = path ?? actual_attachment_name ?? attachment_key` 并断言它恰好命中一个
`atlas_region`；当 attachment 与 region 同名时允许省略 `path`，不因跨 namespace 的合法同名强迫每个
attachment 写冗余 path。

两份 exporter report 只记录各自使用的子集，但必须携带同一个 `global_symbol_table_digest`。MOC3 的
Parts、Deformers、ArtMeshes、Parameters `ids` 数组必须逐项来自该表，`.model3.json`、`.cdi3.json`
和 keyform bindings 只能引用这些已发射 ID。`motion_manifest.json` 的 artifact fragment 同样通过该表
生成；不得写 internal ID 后再由下游猜测清洗规则。缺 symbol、同一 namespace 内重复 export name、
超长、非 ASCII、引用不存在或同一 typed key 得到不同 namespace/name，均以
`missing_export_symbol` 或 `export_name_collision` 失败；不同 namespace 的相同裸名称不得误报碰撞。

### Resume 与失效规则

“文件存在即完成”被拒绝。每个 stage manifest 必须记录：

```text
stage_name
stage_schema_version
algorithm_version
upstream_manifests{stage_name: sha256}
input_file_sha256[]
relevant_config_fingerprint
rig_overrides_sha256
output_file_sha256[]
```

只有 manifest 匹配、所有输出存在且摘要/结构校验通过才可复用。上游 stage、配置、算法版本
或 overrides 任一变化，从第一个受影响阶段向后失效。姿态 provider、模型 SHA-256 和预处理
版本必须进入 joints stage 指纹。`layerdiff/manifest.json`、`optimized/manifest.json`、
`optimized/info.json`、实际 payload 和 canonical-tag registry 版本都进入输入 stage 指纹；
preset/parameter/primitive registry、`RigidDriverRegistry` version/content、
`PrimitiveCandidateEnumerator` version、candidate-universe digest、global symbol-table/
`ExportNamespaceKey` schema 和
capability profile，以及 MaxRects/`CanonicalPngEncoder`/Pillow/zlib version 与设置进入 C stage 指纹；
Spine encoder、Live2D compiler、
MOC3 writer、coordinate schema attestation、parameter map 和各 validator 版本进入对应 export
stage 指纹；G success fingerprint 包含 terminal-finalizer version、profile/validation tier、当前 C/D/E
manifest SHA 和各自 artifact-set digest，G failure fingerprint 则包含相同 finalizer/config 输入、已完成
上游 manifests、有序 `failed_stages` 与规范化 failure-set digest。运行时线程数等不
影响结果的选项不进入指纹。

manifest ownership 还必须满足集合不变量：任意两个 stage 的 `output_file_sha256[].path` 交集为空。
stage manifest 路径按 stage name 保留且不属于 payload output 集合；任何 stage 都不能覆盖另一 stage
的 manifest。
A manifest 只拥有 A cache，B manifest 只拥有 `RigGeometryCache` 等 B cache；C manifest 才拥有
`rig.json/report.json/motion_manifest.json` 与 canonical texture pages，G manifest 独占两个互斥的
terminal artifact。C 修改公共 Rig 不会改变 B output digest；相反，B cache 或 B manifest digest 变化会
通过 `upstream_manifests` 使 C 及 D/E/G 失效。D/E 各记录 C，G 同时记录 C/D/E；map key 按 stage
name 排序后参与 fingerprint。若只损坏
`rig.json` 中的 C-owned 字段，重跑边界从 C 开始，禁止为了重建公共文档重新计算 B 的 mesh/weights。
任何 manifest 声明了另一 stage 已拥有的路径都以 cache-contract error 失败，而不是“最后写入者胜出”。

### 部件规范化与左右状态

入口只构造一次 `CanonicalPartMap: tag → Part[]`。精确 tag 优先，再做受控前缀匹配；同一
canonical tag 的 bbox、mask union 和来源列表都保留，不再“第一个出现者胜出”。

四肢状态明确为：

- `split`：两侧 tag 完整；
- `partial`：只有一侧；
- `merged-separable`：merged mask 有两个可靠连通域；
- `merged-ambiguous`：粘连或组件质量不足；
- `missing`：部件不存在。

`merged-ambiguous` 首版刚性挂到 torso/root，并产生 error 级诊断；只有显式
`--allow-partial` 才允许继续双格式导出。深度用于 draw order、遮挡诊断和 merged 候选排序，
不在已经按语义限定候选骨骼后再重复“禁止后层接受前层骨骼权重”。

### 关节求解：观测、约束、决策

几何求解永远执行，但不承诺永远成功：

```text
parts + masks
  ├─ torso silhouette → shoulder/waist/hip/spine observations
  ├─ head-core/neck contact + head-core axis → head base/top observations
  └─ limb medial-axis graph + inter-part contacts → proximal/distal/end-joint observations
                         ↓
optional pose adapter → pose observations
                         ↓
overrides → authoritative observations
                         ↓
constraint resolver → resolved/unresolved joints + diagnostics
```

所有 `BoneSpec` joint 必须在 A 阶段有显式 observation contract；“export 时再从 bbox 猜一个”
被禁止。`head_core` 定义为 `face` 与可用的 eyes/nose/mouth/ears 核心 mask union，明确排除会被
刘海、长发和帽子拉偏的 `front hair/back hair/headwear`。bbox 只用于尺度归一化和搜索窗口，
不能直接成为 joint 坐标。

| joint | A 阶段主 observation | eligibility / 降级 |
|---|---|---|
| `pelvis`, `spine`, `neck`, `shoulder.*`, `hip.*` | torso 主轴的稳定横截面，加 neck/handwear/legwear 与 torso 的接触带 | 接触缺失或 merged mask 使左右不可分时，对应侧 unresolved |
| `head_base` | `head_core` 与 neck 的接触带中心；有小间隙时取最近边界点对中点 | 最近距离必须小于 `0.15 × head_core_width`，否则只接受 pose/override |
| `head_top` | 从 `head_base` 沿 `head_core` 主 geodesic 轴远离 neck，取投影 95% 分位点 | 不用 hair/headwear 顶端；head_core 太碎或轴方向不稳定则 unresolved |
| `elbow.*`, `knee.*` | limb 主中轴的显著弯折/曲率峰，并由局部宽度与端点方向复核 | 直肢没有显著峰时允许低置信度 length prior，但不能伪装成 mask 观测 |
| `wrist.*` | 中轴上的稳定瓶颈，且瓶颈后必须出现 palm widening/末端分支；pose/override 是独立观测 | v3 `handwear` 通常是整条手臂，平滑轮廓没有腕部信号时必须 unresolved |
| `hand_tip.*` | 从 shoulder 出发的有效 limb 主路径最远端，投影到前景边界内 | 只在独立侧 handwear 且端点分支比通过时创建 |
| `ankle.*` | 独立 `legwear.*` 与 `footwear.*` 的接触带中心；其次才是稳定瓶颈 | 缺 footwear、左右 merged 或接触距离超阈值时只接受 pose/override |
| `toe.*` | 从 ankle 沿 footwear 主轴的最远 geodesic 端点 | footwear 太短、近圆形或方向不稳定时不创建 foot bone |

`length prior` 是带来源标签的弱 observation，只能在直肢且比例落入训练前冻结的区间时参与
resolver；它不能覆盖有效 pose/override，也不能提高为“geometry high confidence”。所有表中
结果连同 mask ID、路径和阈值写入 `joint_observations`，D/E 不得访问 mask 后重算。

A 阶段的指标按可观测性分桶，不用一个总 resolved 率掩盖问题：

- geometry-only wrist **不设最低 recall**；预期大量 unresolved，硬门槛是人工标注集上的
  false-resolve rate `≤5%`。启用 pose 后另报 fused recall，不能混入 geometry 基线；
- ankle 只在左右已拆分、legwear/footwear 均存在且接触距离合格的 eligible 子集上要求
  conditional resolved rate `≥80%`、false-resolve rate `≤5%`；其余样本不进入 recall 分母；
- elbow/knee、hand_tip/toe 也分别报告 eligible count、resolved rate 和误差，不能用肩/髋等
  容易点拉高平均值；
- `wave.*` 需要完整 shoulder/elbow/wrist/hand 链；A/B 可以如实记录 Rig 侧 capability，不能为追求
  动作数量强行制造 wrist。Spine 可消费该 capability；Live2D v1 因 Glue 未验证而逐格式 omitted，
  不能把“骨架可解”误写成“两种 runtime 都可无缝导出”。

几何 confidence 由可解释因素组成：连通性、主路径长度、支路比、端点接触、曲率峰显著性、
mask 内距和左右一致性。每个因素单独记录，避免一个神秘总分。

SimCC raw score 与 geometry confidence 不在同一标度，禁止直接加权平均。首版用确定性决策表：

1. override 合法时无条件优先；
2. geometry 高置信度时采用 geometry；
3. geometry 低置信度、pose 通过自身阈值和 mask/骨长校验时，采用 pose 并投影到合法区域；
4. 两者都有效但差异超过局部肢体宽度阈值时，标记 `unresolved/pose_disagreement`；
5. 部件不存在时不创建相关 joint/bone；存在但求解失败时保留 unresolved，绝不填画布中心。

固定“膨胀 8px”也与分辨率绑定，改成 `max(2px, k × local_limb_radius)` 并设置相对上限。

### 姿态后端：可插拔，SDPose-OOD 优先评测，RTMW-l 备选

接口只返回观测：

```python
PoseBackend.infer(image, person_bbox) -> list[JointObservation]
```

backend 不知道 bone、mesh 或 exporter。输入图固定为 `src_img.png`，前景人物框由身体 mask
union 得到，以版本化的 `bbox_padding=1.25` 外扩，再补到 backend 目标宽高比后做仿射；禁止
直接拉伸任意长宽比、禁止把整张方形 canvas 当人体框，也禁止重合成 amodal layers。
padding、仿射、RGB/归一化和坐标反变换全部进入 joints-stage fingerprint。`src_img.png` 若为
RGBA，v1 先在固定白色 `RGB(255,255,255)` 上做 alpha composite，不能直接丢 alpha 后继承
透明区的黑色 RGB；背景值进入 fingerprint，并在姿态评测证明不合适时随预处理版本升级。

**首选评测候选是 SDPose-OOD Body（17 点）**。选择依据不是只比较同域 AP：官方方法以
Stable Diffusion v2 U-Net 为 backbone，专门评估 OOD 与艺术风格；项目页报告 HumanArt AP
71.2、COCO-OOD AP 63.5，并展示动画/风格化输入。官方 Body 模型卡明确列出 COCO 17 点、
`1024×768 (H×W)` 输入、top-down 工作流和 confidence 输出。17 点已经覆盖首版骨架所需的
肩、肘、腕、髋、膝、踝；Wholebody 133 点不能仅因“更多”就进入首版，只有证明它能减少
脸/手 override 时才增加第二个 provider。

SDPose-OOD 的集成成本也必须如实写出：

- 参数量约 0.95B；官方 Body 仓库当前总计约 5.29 GB，其中 UNet safetensors 约 3.47 GB，
  decoder 约 6.99 MB；
- 当前官方路径是 PyTorch + Diffusers + MMPose，仓库未提供官方 ONNX；依赖清单固定到
  Torch 2.8、Diffusers 0.35、MMCV/MMEngine 等，不能塞进几何基础 extra；
- 它虽然借用 diffusion backbone，但官方 inference 固定 `t=999` 做单 timestep x0 prediction，
  不是多步采样；仍需实测显存、吞吐和冷启动，不能从“diffusion”一词猜性能；
- 官方 Gradio 预处理会把 crop 直接 resize 到目标尺寸。本设计先把 mask union bbox 补到 3:4
  再缩放，避免形体比例失真；必须做与官方脚本的坐标 parity fixture，不能凭肉眼认为等价。

因此 SDPose provider 放在独立 `auto-rig-pose-sdpose` extra，必要时用隔离 worker 避免其严格
Torch/MMCV 版本污染主环境。首版允许用户提供完整本地模型目录；正式下载必须登记为统一
模型源 coherence group，固定文件摘要、模型卡和全部组件，不能只固定 UNet。

**RTMW-l 保留为备选和低成本对照**。它的 Cocktail14 包含 Human-Art，官方报告
384×288 whole-body AP 70.1；DWPose 同尺寸最佳报告 66.5。这个证据比 SDPose-OOD 对艺术域的
直接评测弱，但 RTMW 有可部署 ONNX、体积和依赖明显更小，适合作为显存受限 fallback。
2026-07-31 已直接核对 OpenMMLab ONNX SDK 包：

| 项目 | 已验证值 |
|---|---|
| 上游包 | `rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip` |
| ZIP SHA-256 | `a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd` |
| ONNX 文件 | `end2end.onnx`，229,320,930 bytes |
| ONNX SHA-256 | `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054` |
| 输入 | `input: [N, 3, 384, 288]`，RGB，ImageNet mean/std |
| 输出 | `simcc_x: [N, 133, 576]`，`simcc_y: [N, 133, 768]` |
| body 索引 | COCO-WholeBody `0..16` |

包内 `pipeline.json` 的预处理 `image_size` 仍写成 `192×256`，与实际 ONNX graph 和
postprocess metadata 的 `288×384` 矛盾。因此运行时以 ONNX graph + 本文锁定的模型契约为
准，并在加载时 fail fast；不能盲信 SDK 辅助 JSON。229 MB 也说明原稿“50-100 MB”低估了
成本，但仍远轻于 SDPose-OOD。

RTMW 集成走 `module/onnx_runtime/single_model.py` / `load_session_bundle`，
`bundle_key="auto_rig.pose.rtmw_l_384"`。`module/see_through/model_manager.py` 不负责 ONNX
仓库解析，原稿对此判断错误。正式发布前必须把 ONNX + 模型契约作为一个 coherence group
登记到统一模型源 inventory，包含固定 SHA-256 和 Hugging Face/ModelScope 映射；在映射完成
前只允许用户提供完整本地 ONNX，不新增不受管控的直链下载路径。

SDPose 官方代码/模型卡标为 MIT，RTMW 所用 MMPose 代码为 Apache-2.0；这都不能替代对基础
Stable Diffusion 权重、训练数据和 Cocktail14 各数据集条款的分发审计。发布清单必须记录
每个 provider 的代码、权重、基础模型和训练数据许可证据。

### 声明式骨骼图

单一 `BoneSpec[]` 替代 `needGroup/pivots/parentBone/CREATE_ORDER/connections`：

```text
root: pelvis → spine
torso: spine → neck
neck: neck → head_base
head: head_base → head_top
upper_arm.{side}: shoulder → elbow
forearm.{side}: elbow → wrist
hand.{side}: wrist → hand_tip          （有可靠端点才创建）
thigh.{side}: hip → knee
shin.{side}: knee → ankle
foot.{side}: ankle → toe               （有可靠端点才创建）
```

创建顺序由 parent 拓扑排序得到，诊断连线由 head/tail 得到。骨骼的 `requires` 只引用
canonical part/joint 状态。父骨不存在时允许上提到最近存在祖先，但 head/tail joint 缺失的
骨骼自身不创建。这样不会再生成 pivot 为 `(0,0)` 的“有效骨骼”。

### 图层归属、网格与权重

part 先根据语义得到允许影响它的 bone 集合，再计算权重：头发、脸和衣服不会因为空间距离
接近手臂就被手臂骨骼吸走。`tail/wings/objects` 首版刚性绑定 root/torso 并标记
`dynamic_candidate=true`，不伪造关节链。

网格步骤：

1. mask 清理和连通域拆分；
2. 每个连通分量分别 contour/resample/interior sampling；
3. 空间哈希去重；
4. 每分量 Delaunay；
5. 通过重心、边多点采样和相对外接圆阈值删除跨透明区/狭长三角形；
6. 生成稳定 vertex order、flat triangles、boundary/hull 顺序和局部 UV。

权重步骤：

- 非肢体默认单骨刚性权重；
- 肢体沿主中轴弧长参数化；
- 关节过渡宽度取 `k × distance_transform(joint)`，不使用绝对像素常量；
- 首版通常每顶点两骨影响，但 Rig schema 支持最多四骨；
- 剪除微小影响后重新归一化，验证每顶点权重和为 1；
- 任何 weight 引用都必须指向存在的 bone ID。

依赖不能写成“现有”。当前 `see-through` extra 有 `opencv-python`，没有 SciPy，也没有
`cv2.ximgproc.thinning` 所需的 contrib 保证。新增独立 `auto-rig` extra，明确加入
`scipy`（distance transform/Delaunay）和 `scikit-image`（经过验证的 skeletonize），不要
在首版手写 thinning，也不要为了 thinning 再引入 `opencv-contrib-python` 与现有 wheel 竞争。
姿态依赖拆成 `auto-rig-pose-sdpose` 与 `auto-rig-pose-rtmw`；几何路径不应被数 GB 的
Torch/Diffusers 模型或 229 MB ONNX 强制绑架。

### 共享 TexturePagePlan

Spine 对多 page 宽容，不代表 Live2D 正式包应输出 23-30 张散页。官方 Cubism Web sample
按 `.model3.json` 的 `getTextureCount()` 动态遍历纹理，未发现“最多 4 张”的格式硬上限；但设备
纹理内存、加载时间和下游框架差异足以让一部件一页成为交付风险。因此 deterministic packer
从性能优化提升为正式前置：**A 阶段做精确 rectangle dry-run，C 阶段冻结计划并物化 canonical
pixels/PNG bytes**，而不是
继续留在 E 的待确认项。

`TexturePagePlan v1` 冻结为：

- `2048×2048` 方形 RGBA page，最多 `4` 页；不自动降采样，不自动切换 4096；
- **装箱单位固定为 canonical part payload，不是 mesh 连通分量或 Live2D ArtMesh**。每个
  `optimized/info.json.parts[tag]` 恰好生成一个 region，源矩形就是该 part 的完整 `xyxy` crop；
  透明空洞仍占 atlas 面积。B 阶段产生的多个连通分量，以及未来通过 Glue gate 后可能产生的
  skinning 子网格，都引用同一个 part region，并从原 canvas 坐标推导各自 UV；实验 E0-S 也不得
  借重新裁纹理掩盖几何缝；
- 每 region `padding=2px`、`extrude=2px`、`rotation=false`；
- 使用成熟的 MaxRects BSSF 实现，依赖版本进入 lock/fingerprint；输入先按
  `(max_side desc, area desc, stable_part_id asc)` 排序，所有同分选择再以 page/y/x 排序；
- part 超过可用单页尺寸或四页装不下时返回 `texture_budget_exceeded`，不得退回一部件一页；
- C 按 plan 合成 row-major RGBA pixels，并通过单一 `CanonicalPngEncoder v1` **每页只编码一次**到
  `rig/shared/textures/page_<index>.png`。v1 输入固定为 C-contiguous `uint8 RGBA`，使用项目锁定的
  Pillow + zlib runtime、`optimize=false`、`compress_level=9`，不传入 PNGInfo/ICC/EXIF/DPI，也禁止
  `tIME` 等非确定 metadata；Pillow/zlib 版本、RGBA mode、参数和 metadata policy 全部进入 C
  fingerprint。不承诺跨未锁定 encoder/runtime 版本得到相同字节；
- page、rect、局部 UV、part ID、`rgba_sha256`、`encoded_png_sha256` 和 canonical relative path 写入
  `RigDocument.texture_pages`；两个摘要分别验证像素语义和实际交付字节，不能混成一个含糊的
  “page SHA”；
- D/E 只能以 byte-stream copy 从 canonical path 写各自 page，禁止调用 image decoder/encoder、改变
  metadata 或重新 pack。两份目标文件的 SHA 都必须等于 `encoded_png_sha256`，所以跨 exporter
  逐字节相同是 construction invariant，不依赖两个编码器碰巧采用相同 zlib/filter/chunk 策略。Spine
  atlas 与 MOC3 texture index 只翻译同一份 region plan。

A 阶段拿到 final part bboxes 后，用包含 padding/extrude 的真实 part 矩形运行与 C **同一实现、
同一排序**的 MaxRects dry-run。报告同时写原始 `sum_padded_region_area` 和两个不可混用的比率：

```text
budget_occupancy = sum_padded_region_area / (4 × 2048 × 2048)
used_page_fill   = sum_padded_region_area / (used_page_count × 2048 × 2048)
```

前者衡量固定四页预算消耗，后者衡量实际已用页的装填效率；禁止再输出含糊的
`sum_padded_bbox_area / total_page_area`。报告还包含 used page count、最大 region 和失败原因。
这不是只看总面积的估算：总面积小于容量仍可能因形状装不下。A 失败就停止 item；C 从 A 的
part-region 输入摘要重算并要求 plan 逐字段一致，再写 pixels/UV。B/E 不得因 mesh 拆分改变 region
集合。D/E 不生成 pixels，只校验并复制 C-owned canonical PNG。v1 分辨率白名单下“单 part 超页”分支通常不可达，但作为 schema/未来 profile 的防御检查
保留；真正会触发的主要是四页组合装不下。

4×2048² 是 v1 的产品资源预算，不是假装成 Cubism 格式极限。以后提高页数/尺寸必须作为
profile/version 变更，并在目标 Native/Web/Unity runtime 上重新测峰值纹理内存与加载时间。

### 人工校正

首版不做画布编辑器。`rig_overrides.json` 也是版本化输入：

```json
{
  "schema_version": 1,
  "input_fingerprint": "sha256:...",
  "joints": {
    "joint/elbow.xmin": {"x": 812, "y": 1043}
  },
  "tag_aliases": {
    "objects_2": "handwear-r"
  }
}
```

坐标使用 Rig canvas 空间。fingerprint 不匹配默认拒绝应用，防止把 A 图的 override 套到 B 图。
override 后重新做 mask/骨长校验；确需放在 mask 外必须显式 `allow_outside=true`，并写 warning。
文件摘要变化使 joints 及之后阶段失效。

### 批量动作与表情 preset

“一键”在这里指无交互批处理：每个 item 从 `RigDocument` 推导 capability，套用版本化 preset，
直接写入目标包。它不启动播放器、不截预览、不渲染 GIF/WebM。动作生成和格式编码必须分开：
`PresetBinder` 只产生 exporter-neutral `MotionClip` 和 `ExpressionPreset`；Spine 与 Live2D exporter
只能翻译这些 channel，不能各自重新猜动作、表情或关节。

```json
{
  "id": "clip/idle",
  "preset_version": "motion-core-v1",
  "duration": 2.0,
  "loop": true,
  "channels": [
    {"target_id": "bone/torso", "property": "rotation", "keys": [[0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]}
  ]
}
```

`ExpressionPreset` 是静态目标状态，不伪装成某个格式的动画：它记录 semantic control、目标值、
blend mode 和 capability 等级。Spine 将其编译为同名 animation，Live2D 将其编译为参数 keyform
与 `.exp3.json`。若一个表情只能由一段时间序列表达，它应是 `MotionClip`，不能硬塞进 expression。

正式 dual profile 的 optional 策略冻结为 `optional_preset_parity="per_format"`，不是交集。C 阶段
根据版本化 exporter capability matrix 一次性写 `motion_manifest.json`；D/E 只能执行自己的决策，
不得临时增删 preset。每项至少包含：

```json
{
  "id": "clip/wave.xmin",
  "required": false,
  "supported_formats": ["spine_4_2"],
  "formats": {
    "spine_4_2": {
      "status": "supported",
      "artifact": "spine/skeleton.json#animations/wave.xmin"
    },
    "live2d_moc3_v4_00": {
      "status": "omitted",
      "reason": "live2d_joint_bend_requires_glue",
      "artifact": null
    }
  }
}
```

required preset 的 `supported_formats` 必须与 profile 的 `required_formats` 集合相等，且 duration、
loop、语义 landmark 在两格式间通过 parity；否则 item 失败。optional preset 可以格式不对称，
但 manifest 声明、实际文件和各 exporter report 必须逐项相等；下游不得通过扫描目录猜能力。

首版 capability 与 preset 表：

| 类别 | preset | 最低 capability | 处理 |
|---|---|---|---|
| 常驻动作 | `idle`, `breath`, `body_sway` | resolved root/torso + 有效 mesh | 归一化幅度按 torso 宽高缩放，不写绝对像素 |
| 头部动作 | `head_nod`, `head_shake` | 独立 head/neck bone | 只写 bone rotate/translate timeline |
| 肢体动作 | `wave.xmin`, `wave.xmax` | 对应侧独立 upper-arm/forearm/hand 链与纹理 | Spine 4.2 正式/开发输出可生成；Live2D v1 以 `live2d_joint_bend_requires_glue` omitted，manifest 明确 `supported_formats=["spine_4_2"]` |
| 眼部表情 | `blink` | 闭眼替换层，或可区分的 `eyewhite` + `irides` + `eyelash` mesh | 优先 attachment swap；否则只允许下述 layered procedural blink，禁止压扁整只合成眼 |
| 口型 | `talk` | 独立 mouth 层 + 合法局部 mesh | 周期性局部 deform；没有口腔新纹理时不得宣称音素级 lip-sync |
| 组合表情 | `happy`, `sad`, `surprised` | eyebrow/eye/mouth capability 的声明式组合 | 只组合已有 attachment/deform channel；缺一项则按 profile 失败或省略 |

单张 PSD 只有静态像素，**不包含闭眼、张口或笑脸的新纹理**。因此 expression capability 必须
区分 `native`（有显式替换 attachment）、`procedural`（只靠无三角翻转的 mesh deform）和
`unavailable`。默认不调用生成模型补脸，也不从别的角色借素材；否则批处理会稳定地产出身份
漂移。`rig_overrides.json` 可声明用户提供的替换层别名，但仍要进入 fingerprint。

`procedural blink` 不是对眼部合成图做统一 Y-scale。C 阶段冻结以下分层策略：

1. 在每个可分眼区内，从 eyelash/eyewhite mask 接触与主轴估计 closure line；左右未拆分但有
   两个可靠连通域时允许 coupled blink，不能用一个全局 bbox 横线穿过两眼；
2. `eyewhite` 顶点向 closure line 收拢，闭合帧的可见 alpha 面积必须小于 open 帧的 `5%`；
3. `irides` 随同眼白收拢并在闭合帧 opacity 归零，禁止留下“压扁的瞳孔”；
4. `eyelash` 的上下边界向 closure line 移动，但保持线条局部厚度，闭合帧 alpha 面积须处于
   open 帧的 `70%-130%`，不能把睫毛也压成零高度；
5. 任一层缺失、closure line 不稳定、三角翻转或闭合残留超阈值时，capability 为
   `unavailable`，而不是退回整层 scale。

`talk` 在只有静态 closed-mouth layer 时最多标为 `procedural_silhouette`：可以做小幅开合，但
没有口腔/牙齿/舌头新像素，不能宣称音素级或 native。它和组合表情在默认 profile 中均为
optional；需要稳定表情交付的调用方使用严格 avatar profile，并接受不合格 item 失败。

批处理配置使用显式 profile，而不是“尽量多塞几个动作”：

- `dual_runtime_core_v1` 是正式默认 profile：固定
  `required_formats=["spine_4_2", "live2d_moc3_v4_00"]`，required preset 只有 `idle`、`breath`、
  `head_nod`、`head_shake`，并固定 `optional_preset_parity="per_format"`；blink/talk/组合表情按逐格式
  capability 导出或记录 omitted，`wave.*` 在 Spine 支持、在 Live2D v1 以
  `live2d_joint_bend_requires_glue` omitted；
- `dual_runtime_avatar_v1` 是 opt-in 严格 profile：在 core 基础上要求 `blink`、`talk`，并要求
  `happy/sad/surprised` 达到调用方允许的 `native` 或 `procedural` 等级；它预期会拒绝缺少表情
  素材或 layered deform 质量不达标的 item；
- `spine_4_2_dev` 只用于独立开发/验收 D 阶段，可以产生 `stage_validated` 或
  `stage_validated_with_degradation` 报告，但永远不能写正式 completed `export_manifest.json`；
- `strict_capabilities=true` 时缺任一 required preset 使该 item 失败；false 时允许导出，但
  `motion_manifest.json` 必须逐格式记录 `supported/omitted`、原因、artifact 和 `incompatible_with`。
  正式 profile 的 required preset 不允许通过 `strict_capabilities=false` 降级；optional 的单格式
  omission 不算 partial failure，也不改变 completed 状态。

Spine 和 Live2D 模型文件都不保证加载后自动播放哪个动作。`motion_manifest.json` 可以建议
`default_clip="idle"` 和默认表情，最终 runtime 仍需显式选择。不能把“文件里存在 idle”写成
“会自动播放”。

### Spine 4.2 导出契约

首版只支持精确的 Spine `4.2`，不写模糊的“4.x”。Spine 官方要求 Editor 与 Runtime 的
major/minor 一致，其他版本是后续独立适配器。

exporter 做且只做：

1. 从 `RigDocument.export_symbols` 读取 `GlobalExportSymbolTable` 的 namespaced Spine 子集；禁止在 D
   阶段构造、清洗、改变 namespace 或碰撞消解名称；
2. canvas `+y down` 转 Spine `+y up`；
3. 根据 bone head/tail 计算 parent-local x/y、rotation、length；
4. 把 canvas-space rest vertex 转为每个影响骨骼的 bind-local 坐标；
5. 按官方变长格式写 weighted vertices，扁平化 UV/triangles；
6. 读取 `RigDocument.texture_pages` 写合法 multi-page/multi-region atlas；以 byte copy 物化 page PNG，
   目标 `encoded_png_sha256`、region rect、padding/extrude 与 C 阶段计划完全一致，禁止 exporter 内
   decode/re-encode 或重新装箱；attachment/region 合法同名时
   使用 Spine 默认 lookup，名称不同时显式写 `path`，validator 按 effective path 验证唯一 region；
7. 把 `MotionClip` / `ExpressionPreset` 映射为 Spine `animations` 下的
   bone/slot/attachment/deform timeline；
8. 写 `skeleton.json`、`skeleton.atlas`、pages 和 `export_report.json`；公共
   `motion_manifest.json` 由 C 阶段写一次，不在 exporter 内复制事实源。

骨骼蒙皮、slot draw order、attachment path、
atlas region、required animation 缺任何一项都算导出失败。表情若声明 `procedural`，对应
deform timeline 必须存在并通过无 NaN、索引一致、关键帧拓扑不变和三角形不翻转校验。
默认有 error 级诊断时拒绝导出；`--allow-partial` 只允许已记录的刚性降级，不能绕过格式
validator，状态组合遵循上面的全局枚举。

验证分三层：纯 Python 结构 validator、与官方导出 golden fixture 对比、在有许可证的开发
环境中 opt-in 调用 Spine Editor 4.2/目标 runtime 实际加载并播放 required clips。普通 CI
不假装拥有商业软件。D 只验证并编码 C 阶段已经冻结的 TexturePagePlan，不自行运行 packer。

### Live2D Cubism runtime 导出契约

正式目标固定为 **SDK 可加载的 runtime bundle**，不是 Cubism Editor 工程：

- `.moc3` 使用 header version `3`，即 **MOC3 V4.00**；
- `.model3.json` 使用 `Version: 3`，完整引用 Moc、Textures、Motions、Expressions 和 Groups；
- 同包包含 `.cdi3.json`、PNG 纹理页、`.motion3.json` 与 `.exp3.json`；
- 不把它叫“Live2D 4.2”。Spine `4.2` 与 Cubism MOC3 `V4.00` 是两条独立版本轴。

选择 V4.00 是为了覆盖本需求所需的 ArtMesh、parameter/keyform 和 motion/expression，同时保持
Cubism 4.0+ runtime 的较宽兼容面。只有当真实功能依赖 V4.02/V5 字段时才升级版本，且升级必须
同时提高最低 Cubism Core 版本并重跑 release gate，不能只改 header byte。

实现放在 `module/auto_rig/export/live2d/`，使用隔离的原生 Python compiler/writer，不引入
Electron、浏览器或 Node 运行时。StretchyStudio 固定 commit 的 MIT 代码可作为字段布局和测试
证据，复用时保留许可证与来源；禁止把 230 KB 的 `.cmo3` writer 整体移植进来。exporter 先从
`RigDocument` 编译一个可丢弃的 `CubismDocument`，该对象不是新的公共事实源，也不得反向修改
Rig。

compiler 做且只做：

1. 先读取 `RigDocument.export_symbols` 的 namespaced Live2D 子集，再把 part 层级映射为 Cubism Parts、把 mesh
   连通分量映射为一个或多个 ArtMeshes；Parts/Deformers/ArtMeshes/Parameters ID 以及 model3/cdi3
   引用全部使用全局映射，禁止 E 阶段重新清洗。保留稳定 draw order；同一 part 派生出的全部
   ArtMesh 共享该 part 的 atlas region，不重新裁纹理；
2. 从 canvas-space Rig 构建版本化 `Live2DCoordinatePlan`。PPU 归一化只用于 canvas→root model
   边界；每个 WarpDeformer、RotationDeformer 和 ArtMesh 都按其**直接父节点**转换到对应 local
   frame，禁止把同一 root 公式套到整棵树。逐层 forward/inverse、父 ID、frame kind、输入摘要和
   round-trip residual 写进报告；
3. 从 `RigDocument.texture_pages` 确定性分配 texture index 与 UV，并把 canonical PNG 原样复制到
   Live2D texture path；页数必须 `≤4`、每页必须 `2048×2048`、目标 SHA 必须等于
   `encoded_png_sha256`，且所有 texture reference 能从 `.model3.json` 闭环解析；
4. 只在 **ID 命名层**优先使用标准参数：`head_shake→ParamAngleX`、
   `head_nod→ParamAngleY`、`body_sway→ParamBodyAngleX`、`idle→ParamAutoIdle`，以及适用的
   `ParamAngleZ`、`ParamBreath`、
   `ParamEyeLOpen/ROpen`、`ParamMouthOpenY`、`ParamMouthForm` 和 brow 参数；耦合眼部或无法对应
   标准含义时使用稳定 ASCII custom parameter，并在 `.cdi3.json` / report 中说明。标准 ID 只改善
   EyeBlink/LipSync group、面捕和第三方 runtime 的语义兼容，**没有任何内建变形行为**；core v1
   的 `head_nod/head_shake` 仍只是固定的 2D pivot/translate 风格化动作，不得因使用
   `ParamAngleY/X` 就宣称重建了新视角、3D yaw/pitch 或可泛化的面捕行为；
5. 先从 `motion_manifest` 的 Live2D-supported channels 解析确定性的 `Live2DBindingPlan`，再由该计划
   构建 `Live2DDriverLiveness`；RotationDeformer 实例键固定为 `(bone_id, parameter_id)`，不是 bone
   单键。只把活性实例子图编译为嵌套 RotationDeformer，不得把完整 Rig bone 树机械复制进 MOC3。
   骨骼旋转、pivot
   平移和统一缩放写入 `angles/origin_xs/origin_ys/scales` keyforms，live 子 bone deformer 的
   `parent_deformer_index` 指向最近 live 父 deformer，刚性 ArtMesh 指向最内层有效 deformer；多个独立
   刚体参数作用于同一区域时，按 `rotation-stack-v1` 的 `stack_rank` 组成嵌套 deformer stack，不展开
   成顶点采样表；v1
   对其余三个字段固定 `opacities=1`、`reflect_xs=false`、`reflect_ys=false`，不支持的反射动作必须
   在 capability 阶段拒绝，不能留下未初始化 section。同一节点同时改变 angle/origin 时统一使用
   `p_parent(t)=origin(t)+R(theta(t))·(scale(t)·p_local)`，其中 `theta` 对 MOC3 `base_angle` 的换算
   由 E0-core 固定；不能在 exporter 内交换旋转/平移顺序；
6. 把 `breath` 等非均匀结构形变编译为单参数 WarpDeformer，把对应 RotationDeformer 链作为其
   子级；任意非均匀 scale/shear 都不得伪装成只有单一 `scales` 字段的 RotationDeformer；
7. 正式 v1 只对 breath/其他结构 warp、blink/talk、表情和其他真正改变局部形状的已支持 channel
   烘焙 ArtMesh/warp keyforms 或 opacity binding；LBS 混合带只能由 E0-S 实验 compiler 产生。第 4 条
   决定参数名，第 5-7 条决定原语和 binding；
   `.motion3.json` 只负责驱动已经绑定的参数，不能替代任何 binding；
8. 把 `blink`、`talk` 和其他时间序列写为 `.motion3.json`；把 `happy/sad/surprised` 写为
   `.exp3.json`。需要网格变化的 expression 同样先生成 parameter keyforms，`.exp3.json` 只写
   参数值和 `Add/Multiply/Overwrite` blend；
9. 在 `.model3.json` 注册 motions、expressions，以及实际存在的 `EyeBlink` / `LipSync` 参数组；
10. 写 `.moc3`、JSON、纹理、参数映射和 `export_report.json`，不得声明未生成的 physics/pose 文件。

#### Live2D deformer 活性剪枝

先把公共 `motion_manifest.json` 中 `formats.live2d_moc3_v4_00.status="supported"` 的每个
channel/expression target 解析成 `Live2DBindingPlan`。每个 supported channel 必须恰好对应一个计划项；
计划项至少记录 `preset_id/channel_id/rig_target_id` 和非空
`bindings=[{candidate_id, parameter_id, primitive_kind, primitive_target_id, stack_rank}]`。每个 binding
必须是 `RigDocument.primitive_candidates` 中一个 immutable record 的实例化结果。RotationDeformer 的
`primitive_target_id` 必须引用 typed `(bone_id, parameter_id)` instance key，不能只写 bone ID；
`stack_rank` 对非 rotation primitive 为 `null`。一个 channel 可以合法驱动多个 ArtMesh/opacity target，
但不能有两份互相矛盾的计划项；无 binding 时先报
`live2d_parameter_binding_missing`。计划只由 Rig、preset 和版本化原语选择规则推导，**不得读取 writer
已生成的 MOC3 section 再反推活性**，否则会把“先决定是否发射节点”和“发射后才知道 binding”做成
循环依赖。

这里的 pair 使用 Rig bone internal ID 与 parameter registry internal ID；清洗后的 MOC3 export name
不是身份键。instance 的导出 ID 只能由 `GlobalExportSymbolTable` 从 typed pair 派生，不能反过来用
字符串拆分恢复 bone/parameter；`primitive_target_id` 不在 C 阶段 candidates/symbol table 中时必须
`missing_export_symbol` 失败。builder 不暴露接收裸 bone/parameter 并创建 candidate 的 API。

同一 `(bone_id, parameter_id)` 被多个 clip/expression 使用时只生成一个 RotationDeformer instance，
前提是它们声明完全相同的 rest frame、keyform mapping 与 `stack_rank`；不一致时报
`live2d_parameter_conflict`。v1 即使 runtime 格式技术上允许一个 deformer 绑定多个普通参数，也
**不使用**该路径：官方 Editor 对一个对象映射多个普通参数会形成参数 keyform 组合，和本设计拒绝
多参数全量 grid 的决定冲突。

`rotation-stack-v1` 只冻结排序语义：lower rank 为外层、higher rank 为内层。production 行内容属于
独立的 `RigidDriverRegistry v1`：

| semantic driver | parameter ID | 同 bone `stack_rank` |
|---|---|---:|
| `body_sway` | `ParamBodyAngleX` | 100 |
| `idle` | `ParamAutoIdle` | 200 |
| `head_shake` | `ParamAngleX` | 300 |
| `head_nod` | `ParamAngleY` | 400 |

registry load 与普通 CI 必须断言**所有 built-in rigid driver 的 rank 全局唯一**，不能因为当前 torso/
head 恰好不同 bone 就复用数字。rank 的比较仍只发生在同一 bone；bone ancestry 先决定外层父骨与
内层子骨。首个 same-bone instance 的 parent 是最近 live parent bone 的最内层 instance，后续
instance 依 rank 串接，ArtMesh 挂到本 bone 最内层 instance。新增 rigid driver 必须分配全局未占用
rank；built-in 重复值必须使普通 CI 失败，worker 的 defense-in-depth 校验也必须在枚举 item 前以
job-level `invalid_rigid_driver_registry` 非零退出，不创建 staging 或 `rig/report.json`。它不是
`live2d_deformer_order_conflict`：后者只用于某个已存在 item 的 binding plan 对同一 typed deformer
instance 提供互相矛盾的 rank/order 数据。两种情况都不允许用 parameter 字典序兜底。

`live2d_frame_contract_digest` 只包含 `rotation-stack-v1` 的 comparator、lower→outer 语义、冲突行为、
parent chaining 规则和使用 synthetic ranks 的非交换 E0 结论，**不包含**上表 production rows 或其
摘要。`RigidDriverRegistry` version/content digest 进入 C/motion stage 与 D/E export cache fingerprints；
新增 driver、改 rank 或改 parameter mapping 必须重新生成 candidate set、symbol table、binding plan 和
输出，但不需要重新签署 E0。`export_report.json` 同时记录 stack-semantics version 与 registry digest，
不能再用一个模糊的 `rotation-stack digest` 混合两层。

`Live2DDriverLiveness` 只消费这份 binding plan，以及这些 target 必需的结构 WarpDeformer；
Spine-only optional preset 不能让 Live2D bone 变活。算法冻结为：

1. 以 binding plan 中实际绑定到 Live2D parameter 的 rotation instance、warp、ArtMesh/opacity target
   为 live seeds；
2. 直接有 driver 的 `(bone_id, parameter_id)` instance 才发射 RotationDeformer。未驱动祖先 bone
   不作为 pass-through deformer 保留，
   而是把其 rest parent-local transforms 按原顺序合成到最近 live 祖先与 live child 之间；
3. 只挂在被剪枝 bone 下的静态 ArtMesh，把完整 rest transform 烘入最近 live parent/root 的 local
   coordinates 后重新挂接；不得改变 canvas-space rest pose、UV 或 draw order；
4. `Live2DCoordinatePlan` 只基于剪枝后的实际 parent graph 构建；不得先按完整 Rig 树算 local frame
   再删除中间节点；
5. `export_report.json` 分别记录 binding-plan/candidate-universe/rigid-driver-registry digest 与
   rotation-stack-semantics version；`deformer_liveness` 再逐 typed
   deformer instance 记录 `emitted/pruned`、bone/parameter/driver IDs、stack rank、旧/新 parent、
   composed rest transform 和受影响 ArtMesh。任何 emitted deformer 没有非空 parameter/keyform
   binding 都返回 `live2d_dead_deformer`。

这不是一张手工 anatomy 白名单。默认 core 的典型结果如下，但最终集合必须由 driver graph 得出：

| Rig role | Live2D v1 的典型 instance pairs |
|---|---|
| root | 只作为 root frame/Part，不创建 RotationDeformer |
| torso | 按实际 channel 分别发射 `(torso, ParamBodyAngleX)`、`(torso, ParamAutoIdle)`，不是一个 torso node |
| neck/head | 按实际 target 分别发射 `(bone, ParamAngleX)`、`(bone, ParamAngleY)`；未驱动的中间 neck 可折叠 |
| upper_arm/thigh | 只有版本化 preset 明确包含独立 parameter channel 时才发射对应 pair，不能因“可能以后会动”保留 |
| forearm/shin/hand/foot | Live2D v1 没有 supported joint-bend preset 时剪枝；Spine 的 `wave.*` 不改变此结论 |

剪枝 validator 必须同时证明：所有 emitted deformer 从至少一个 Live2D parameter binding 正向可达；
所有 Live2D-supported channel 的 target 均反向解析到一个 emitted primitive；剪枝前后所有参数处于
default 时的 canvas-space rest vertices 相差 `≤0.1 px`。

#### Live2D 坐标空间契约

`RigDocument` 继续只保存 LayerDiff canvas pixel。`Live2DCoordinatePlan v1` 是 E 阶段的确定性编译
产物，根边界先做且只做一次：

```text
PPU = max(canvas_width, canvas_height)
canvas_to_root(x, y) = ((x - canvas_width/2) / PPU,
                        (y - canvas_height/2) / PPU)
```

预期的 MOC3 frame schema 冻结为下表，但必须先通过 E0-core 实载才能把
`coordinate_schema_version="live2d-frames-v1"` 写入 `export_report.json` 和 export manifest
fingerprint；不得向 MOC3 私加未定义 metadata section：

| 节点/数据 | 编码 frame | 规则 |
|---|---|---|
| root ArtMesh/keyform | `ROOT_MODEL` | 使用上面的 PPU-normalized root model 坐标 |
| WarpDeformer grid keyform | 直接父节点的 output frame | root 父节点用 `ROOT_MODEL`；warp 父节点用父 warp local；rotation 父节点用父 rotation local |
| WarpDeformer child input | `WARP_LOCAL` | 预期输入域为 `[0,1]×[0,1]`；由 rest grid 的 inverse map 从父 frame 求得，不能用 part bbox 线性归一化替代一般逆变换 |
| RotationDeformer origin | 直接父节点的 output frame | 父为 warp 时预期是 warp local；父为 rotation 时预期是相对父 pivot 的 local offset；禁止直接写 canvas/ROOT_MODEL 值 |
| RotationDeformer child input | `ROTATION_LOCAL` | 预期为相对该 rotation pivot 的 local 坐标；angle 与 scale 无量纲，但它们的最终空间效果随父 frame 组合 |
| ArtMesh keyform | 直接父 deformer 的 input/local frame | 无父节点才允许 `ROOT_MODEL`；挂在 warp/rotation 下时不得再次套 root PPU 公式 |

这里的 warp-local `0..1` 与 rotation-local 规则来自 StretchyStudio 对 CMO3 Editor 5.0/Hiyori 的
逆向记录，不冒充已公开的 MOC3 官方规范。该项目现有 MOC3 writer 只发射 root-level warp，并把
位置统一 PPU-normalize，无法验证嵌套 runtime 行为。E0-core 必须至少覆盖
`root→warp→rotation→rotation→ArtMesh`，在 rest、单层驱动和多层同时驱动时读取最终 runtime
vertices；若观测与上表不符，停止 E0-core 并修订 schema/spec，不允许按数值范围猜 frame、静默
回退到单一坐标变换或签署 attestation。

E0-core 证明的是**窄的 frame/runtime 语义契约**，不是“这份 compiler 源码每一行都被认证”。
`live2d_frame_contract_digest` 对 canonical descriptor 求 SHA-256，descriptor 只包含：

1. `coordinate_schema_version`、frame table 和 canvas/root/warp/rotation/ArtMesh frame kinds；
2. 只承载 `encode_point/decode_point`、`T·R·S`、`rotation-stack-v1` comparator/parent-chaining 语义和
   相关 binary field codec 的 side-effect-free semantic kernel 模块源码摘要、版本、语义 descriptor
   与 synthetic 纯函数测试向量；production `RigidDriverRegistry` rows 明确排除；
3. 影响 deformer parent、keyform binding、origin/positions 的 MOC3 header/SOT section-layout descriptor
   版本与摘要；
4. E0 fixture/golden SHA-256、default-rest/round-trip/runtime parity 结果；
5. 逐个通过同一 E0 矩阵的 approved Core binary SHA-256 allowlist，以及执行 Core call/assertion 顺序的
   `e0_validator_protocol_digest`；不把 harness 的日志/报告实现纳入 gate。

semantic kernel 必须是隔离的小模块，禁止日志、报告渲染、CLI、错误文案和业务 orchestration；这些
非语义代码留在外层 compiler/writer。这样 kernel 的任意实现变化都会失效 attestation，而外层无关
改动不会。不能为了少跑 E0 把实际坐标/layout 逻辑移出 kernel。

E0-core 通过后生成并随 compiler package 固定
`module/auto_rig/export/live2d/attestations/live2d-frames-v1.json`。它包含上述 gate descriptor/digest，
另有不参与 coordinate gate 的 `provenance`，记录完整 compiler/writer version、源码摘要、构建
commit、完整 SDK harness 源码摘要和生成时间。完整源码摘要仍进入 export cache fingerprint 和报告，
但改日志、增加 report 字段或修复与 frame 无关的错误路径，不会仅因 provenance 变化强迫重跑
E0。反过来，改 frame codec、transform/
stack order 或相关 layout descriptor 必须改变 contract digest；普通 CI 还要运行 attestation 内的纯函数
向量与 parser golden，防止实现变了却漏升 descriptor 版本。

这里的“签署”是可复现的摘要绑定，不额外引入 PKI，也不是每个 item 的产物。扩大 Core 兼容性不能
只写 semver range：每个新增 Core binary 都必须按相同 validator protocol 实际通过 E0，加入
allowlist 后生成新 attestation。

启动 gate 按以下顺序执行，且都先于输入 item 枚举：

1. 所有 structural/release job 先加载并验证 packaged preset/parameter/primitive/
   `RigidDriverRegistry`；其 built-in rank 重复、缺必需字段或 canonical ID 冲突均返回 job-level
   `invalid_rigid_driver_registry`，且不创建 item staging/report；
2. 所有 structural/release job 重算本地
   `live2d_frame_contract_digest` 并执行纯函数向量；attestation 缺失、descriptor/digest/vector 不匹配时
   返回 `live2d_coordinate_schema_unverified`，且不创建 item staging；
3. release tier 还必须确认配置的 Core binary 位于 attested allowlist，且本地 SDK
   harness 声明并通过相同的 `e0_validator_protocol_digest`。
   缺失时报 `live2d_release_gate_unavailable`，存在但未经 E0 attested 则报
   `live2d_core_unattested`。普通 CI 的 structural tier 不需要 Core，也不生成 attestation；它可以跑完整
   compile/parser/golden 流程并写 stage report，但永远不能写正式 completed `export_manifest.json`。

运行期 item 只执行自己的 coordinate plan round-trip；其失败码是
`live2d_coordinate_roundtrip_failed`，不能借 startup 错误混淆。attestation 也不替代 release tier 对每个
正式 MOC3 的 consistency、实载与动作验证。

plan 中每个节点至少记录 `node_id/type/parent_id/input_frame/output_frame`、rest grid/pivot、
`encode_point()` / `decode_point()` 的版本、输入摘要和测试点 residual。warp 映射是网格变换，不能
伪装成一个 3×3 矩阵；对全部 mesh vertices、warp control points、rotation origins 及额外边界采样点
执行 `canvas→encoded local→解析 forward stack→canvas`，最大 round-trip 误差必须 `≤0.1 px`。
超阈值返回 `live2d_coordinate_roundtrip_failed`，不得靠增加 non-rigid stops 掩盖 frame 错误。

原语选择冻结为：

| Rig 语义 | Cubism v1 原语 | 误差/预算 |
|---|---|---|
| bone rotation、pivot translation、uniform scale | 嵌套 RotationDeformer | 运行时解析变换；不计顶点弦割误差或 baked-position 预算 |
| `breath`、非均匀 scale/shear、区域级结构形变 | 单参数 WarpDeformer | 对 control points 做数值采样与局部宽度误差门 |
| 两骨/多骨 LBS 过渡 | **正式 v1 不编译**；E0-S 可实验 blend band，未来正式路径优先 Glue | 实验逐顶点误差与 seam 测量；不产生 v1 capability |
| blink/talk/表情 | ArtMesh keyforms + opacity | 逐顶点、面积残留和三角翻转门 |

`Live2DSkinningPartition` 改名为 `ExperimentalLive2DSkinningPartition`，只属于 E0-S，不回写 Rig、
不进入正式 exporter。它可以把权重为 1 的三角形归入对应 bone 的刚性 ArtMesh，把跨 bone 三角形
归入 baked blend band，用来量化问题，但不能宣称无缝：若刚性边界和 blend-band 边界在 stops 上
重合，stop 之间前者仍走旋转圆弧、后者走线性弦；对 `ρ>0, Δθ≠0`，中点至少存在
`ρ×(1-cos(|Δθ|/2))` 的几何偏差。atlas region 的 extrude 只处理外边缘采样，不能修补 region 内部
的这条缝。

因此正式 dual-runtime v1 在 Live2D 侧对 `wave.*` 固定记录 `live2d_joint_bend_requires_glue` 并
omitted；Spine 侧可以正常生成。这是 `optional_preset_parity="per_format"` 明确允许的 optional 能力
分叉，必须由公共 manifest 声明，不能影响 required preset 的双格式原子性。E0-S 的
`live2d_skinning_seam` 固定为实验 warning，只影响 Live2D optional preset，不得让 core/avatar item
或 E0-core 失败。Cubism Editor 原生 Skinning 会拆 ArtMesh 并用 Glue 连接；未来若新增要求独立关节
弯曲的正式 profile，必须先把 Glue section、权重、keyforms 和官方 runtime parity 纳入 E0 硬门。
另一条可接受但必须另行修订 spec 的路径，是定义非零 render-space seam 阈值并同时给出确定性的
遮盖/重叠策略；不能继续把 `E_i≤0.05×r_i` 与“零裂缝、零重叠”写成同时成立。

RotationDeformer 先在线性 keyforms 间插值 angle/origin/uniform scale，再在运行时对孩子执行变换；
因此刚性 rotation 不走顶点直线，也没有下述弦割收缩。线性参数映射通常只需 min/max 两个
keyforms，但前提是它们在 parameter default 处的 runtime 插值**严格还原 rest form**；否则必须
增加显式 default keyform。default 等于端点或两端按构造可证明插值回 rest 时不重复加 key。
这由每个模型的 `default_rest_invariant` validator 决定，不等待 runtime 报错，也不允许为了掩盖
错误坐标系而 adaptive 加点。每个单 deformer 与嵌套 deformer chain 都要在参数区间取
9 点，与解析 2D transform chain 比较，最终 canvas-space 顶点误差必须 `≤0.1 px`。

下面的误差契约**只适用于 baked ArtMesh/warp 的非刚性位置**：正式 v1 的 breath/表情形变，以及
E0-S 或未来 Glue 路径中的 joint blend。对实验 joint blend band，相邻 stops 的单骨旋转跨度
`Δθ`、距 pivot 为 `ρ` 时，顶点弦割误差上界为：

```text
e = ρ × (1 - cos(|Δθ| / 2))
```

多骨影响使用 `Σ(weight × e)`，父子 transform chain 用三角不等式累加各 link 的保守上界；
非刚性 scale/shear、warp、blink/talk 或无法证明的组合必须用真实数值采样复核。non-rigid compiler
对每个参数区间 adaptive subdivision，直到每个受影响顶点 `i` 的保守上界与 1/4、1/2、3/4
数值采样实测误差都满足：

```text
E_i <= 0.05 × r_i
```

`r_i` 是顶点 `i` 在**自身 deform region 与连通分量内**最近的有效 medial-axis sample 上的
distance-transform 半径，不是整段中位数，单位为 canvas pixel；不允许借用邻接图层或另一分量
的粗半径。找不到合法 sample 时该 region 不能通过 keyform compiler。顶点越靠近细腕、脚尖或
发梢，允许误差越小并自动触发更多 stops。
`ρ_i` 是长度量级、`r_i` 是宽度量级并非单位错误：用局部宽度归一化是刻意的视觉误差契约，
用于防止细长远端被粗段的宽度预算放过。golden fixture 还要在每个最终区间取 9 个等距参数值，
逐顶点比较 Cubism 线性插值与真实 LBS。

v1 的 `17` stops、`total_baked_vertex_positions ≤ 1,000,000` 只统计 ArtMesh/warp 的 non-rigid
keyform positions，不统计 RotationDeformer 的七个标量 keyform 字段；`.moc3 ≤ 64 MiB` 仍约束
整个文件。任一 optional non-rigid preset 在上限内达不到误差阈值就 omitted；required preset 则
以 `live2d_interpolation_error` 或 `live2d_keyform_budget_exceeded` 失败，不能偷偷降低 stop 数。
这三项在 report 中统一标为 `capacity_guard`，只用于防止异常编译规模和内存放大，不得进入质量
分数、模型排序或“形变更好”的判断。required `breath` 会贡献少量 warp control-point positions，
所以默认值不是严格为零，但通常远离上限；外部纹理页完全不计入 `.moc3` 字节数，由
TexturePagePlan 的独立预算负责。

首版明确 **不生成同一 ArtMesh 的多参数全量 keyform grid**。`live2d_parameter_conflict` 只在两个
non-rigid driver 直接修改同一个 ArtMesh/warp 且无法按 joint band、语义区域或连通分量拆开时
触发。父子 RotationDeformer、WarpDeformer → RotationDeformer、RotationDeformer → 单参数
ArtMesh 等层级组合不是 conflict；它们由 runtime 顺序组合。默认 core profile 固定使用可组合
层级：`breath` 走结构 WarpDeformer；`body_sway/idle` 分别生成 rank 100/200 的 torso
RotationDeformer instances；`head_shake/head_nod` 分别生成 rank 300/400 的 head-target
RotationDeformer instances。因此不得再因它们影响同一纹理而返回 `live2d_parameter_conflict`，
但 instance key 或 rank 不一致必须分别报 parameter/order conflict。

StretchyStudio runtime writer 只证明 V4.00 的基础 sections 与 WarpDeformer 可被 Cubism Viewer
5.0 / Ren'Py 读取；它虽然声明 RotationDeformer section layout，当前 runtime 数据构建没有生成
rotation deformer hierarchy，也没有 `.exp3.json`。因此本实现不能把“移植 writer”计作 E 阶段
完成。E 阶段必须新增 RotationDeformer、parent binding、逐层 `Live2DCoordinatePlan` 与 expression；
正式 v1 不以未验证的局部 skinning keyforms 代替 Glue。
验证明确分成两级，而不是让商业/原生依赖污染普通 CI：

1. **普通 CI 必跑**：纯 Python 结构/parser、schema、deterministic bytes 和官方导出 golden
   fixture；没有 Cubism Core 不能跳过这些测试；
2. **opt-in release gate**：对每个正式输出调用 `csmHasMocConsistency`，再由官方 Cubism SDK
   或 Viewer 实际加载、非空渲染并逐项播放 motion/expression。普通 CI 未配置 Core 时显示带原因
   的 skip，不置红；E0-core、release job 或正式批处理缺 Core/SDK 时则报
   `live2d_release_gate_unavailable`、返回非零且不写 completed manifest。

参数名本身不产生任何模型行为。结构 validator 还必须证明每个被 motion/expression 驱动且声明
可见效果的标准或 custom parameter 至少绑定一个非空 RotationDeformer、WarpDeformer、ArtMesh
keyform 或 opacity target；官方 runtime 再证明非默认参数确实改变 deformer、顶点、opacity 或
最终像素。反向也必须成立：每个 emitted Rotation/WarpDeformer 至少被一个 Live2D-supported
parameter binding 直接驱动；仅为保持完整 Rig 树而存在的静态 pass-through deformer 是
`live2d_dead_deformer`，不能进入 writer。

### 错误与诊断

job-startup gate 使用独立的 job-level result，不写进任何 item 的 `rig/report.json`：

```text
invalid_rigid_driver_registry
live2d_coordinate_schema_unverified
live2d_release_gate_unavailable
live2d_core_unattested
```

这些错误在枚举 item 前返回非零，因此没有 item/part/joint ID，`continue_on_error` 也不适用。v1 的
`RigidDriverRegistry` 是 packaged/versioned registry，不接受临时用户扩展；若未来开放扩展，它仍必须
先合并并通过同一 startup validator，不能把配置错误下放成逐 item 失败。

`rig/report.json` 只保存已经枚举出的 item diagnostics，使用稳定代码而非只能搜索的自由文本，至少包括：

```text
input_contract_mismatch
stage_artifact_ownership_conflict
postprocess_info_required
raw_part_source_forbidden
unknown_tag_version
unsupported_non_square_frame
unsupported_auto_rig_canvas_resolution
missing_part_payload
psd_bbox_mismatch
merged_limb
partial_limb
rigid_fallback_applied
unresolved_joint
pose_rejected_by_mask
pose_geometry_disagreement
degenerate_mesh
invalid_weight_sum
export_name_collision
missing_export_symbol
missing_required_capability
invalid_motion_clip
invalid_expression_preset
texture_budget_exceeded
spine_validation_failed
live2d_parameter_conflict
live2d_deformer_order_conflict
live2d_parameter_binding_missing
live2d_dead_deformer
live2d_keyform_generation_failed
live2d_coordinate_roundtrip_failed
live2d_default_pose_mismatch
live2d_joint_bend_requires_glue
live2d_skinning_seam
live2d_interpolation_error
live2d_keyform_budget_exceeded
moc3_consistency_failed
live2d_expression_application_failed
live2d_validation_failed
dual_export_incomplete
```

在 C 成功前，同一诊断 record schema 写入 owner stage 的私有 report/manifest；A/B 失败不会为了产出
错误信息而创建一个不完整的公共 `rig/report.json`，而是由 G 把规范化 failure record 发布到公共
`rig/error.json`。C 成功时才把 A-C diagnostics 快照到其 owned report，之后 D/E 仍只写各自 export
report；任一 item failure 最终也由 G 更新同一个公开错误契约。

每项带 `code/severity/stage/item_id/repair`；只有错误确实归属于某个实体时才带 typed
`entity_ref={kind,id}`，不能为 item-global export/cache 错误伪造 part/joint ID。正常模式出现 error 返回非零；
批处理的 `continue_on_error` 只隔离 item，不能把失败 item 记成 completed。
`live2d_joint_bend_requires_glue` 在正式 v1 固定为 `warning + preset_omitted`；
`live2d_skinning_seam` 只允许出现在 E0-S，固定为 experimental warning。当前没有 profile 把
独立肘/膝弯曲列为 required，因此两者都不得升级为 item error；未来 profile 若要求该 capability，
应在 profile 选择时以 `missing_required_capability` fail fast，直到 Glue E0 gate 已被版本化启用。
`live2d_coordinate_schema_unverified` 是 job-startup error，只表示 attestation 未生成、
frame contract descriptor/digest 或纯函数向量不匹配；它不比较完整 compiler/writer provenance。
release tier 缺 Core/SDK 用 `live2d_release_gate_unavailable`，存在但不在 E0 allowlist 用
`live2d_core_unattested`。built-in rank/registry schema 不合法只用
`invalid_rigid_driver_registry`。`live2d_coordinate_roundtrip_failed`、`live2d_default_pose_mismatch` 和
`live2d_deformer_order_conflict` 才是具体 fixture/item 的编译、binding 或实载错误。

### 集成点与实施顺序

- CLI：`module/auto_rig/cli.py`，复用项目批处理、Rich progress 和 continue-on-error UX；但
  `--skip_completed` 必须调用 auto-rig `StageGraphValidator`，不得复用 see-through
  `detect_resume_stage()`/“最终文件存在即完成”的 predicate。每个已枚举 item 的失败通过 G 写
  `rig/error.json`，与现有批处理可寻址错误记录的产品习惯一致；
- 配置：`config/model.toml` 新增 `[auto_rig]`，不修改 `[see_through]` 默认值；正式默认
  `profile="dual_runtime_core_v1"`，其 `required_formats` 固定为 `spine_4_2` 与
  `live2d_moc3_v4_00`；
- 验证层级：产品批处理固定 `validation_tier="release"`；测试 harness 可显式使用
  `validation_tier="structural"` 跑完整 compile/parser/golden 和 attestation vector，但只能写 stage
  report，不能产生 completed `export_manifest.json`；层级不改变 capability/profile 语义；
- GUI：`gui/wizard/step6_tools.py` 新增工具 tab，走 `job_manager.submit()` +
  `ExecutionPanel`；只负责提交批任务和显示报告，不建设 canvas/player；
- 脚本：`2.6.1.auto_rig.ps1`；
- 包装：`pyproject.toml` 增加 `auto-rig`、`auto-rig-pose-sdpose`、
  `auto-rig-pose-rtmw` extras；MOC3 writer 本身不引入 JS runtime，官方 Cubism Core/Viewer 只作为
  opt-in release validator 探测，不随 Python wheel 擅自再分发；已通过 E0-core 的
  `attestations/live2d-frames-v1.json` 作为 package data 分发，structural/release worker 启动时先验证
  frame contract，release worker 再验证本机 Core allowlist；
- 模型：SDPose 完整组件组与 RTMW ONNX 各自登记 coherence-group inventory；姿态开关只有在
  对应 provider 自检通过时才可用。

“对现有代码改动为零”不成立。虽然不改 see-through 产物行为，仍需修改 config、GUI、
packaging、模型 inventory 和测试注册。实现计划必须按 A-F 分提交，不能用一个巨型 PR 同时
落地几何、preset、Spine、Live2D 和姿态模型；第一份实现提交应是隔离的 E0-core fixture/writer
feasibility gate，而不是先把整个 auto-rig 写完再验证 MOC3 能否交付。E0-S 必须是独立实验 job，
不得因为无 Glue seam 不通过而阻塞 E0-core。

---

## 兼容性风险

| 风险 | 影响 | 处理 |
|---|---|---|
| canvas 被误当成原图像素 | override/导出坐标无法映射，宽高比判断错误 | v1 冻结 LayerDiff canvas；不提供未经持久化的原图反变换 |
| 768/1024/1280 letterbox 后主体过小 | 细肢、耳朵、发梢的几何不稳定 | mask union bbox；按分辨率/长宽比分层评测；低于阈值 unresolved/刚性绑定 |
| `tag_version` 未从正确 manifest 读取 | canonical requires 永不满足或误配纹理 | 强制读取 `layerdiff/manifest.json`；registry 按 v3；未知版本失败 |
| 根/optimized 两个 `info.json` | 读错时有 tag 但没有几何，可能静默产空 Rig | 只接受 `optimized/info.json`；严格字段/schema 诊断 |
| 根目录 PNG 与最终 tag 宇宙不同 | split 后仍误用 merged 纹理 | 根 PNG 明令禁用；PartSource 仅支持 final PSD 或 optimized PNG |
| `frame_size` 的 H/W bug 被方形掩盖 | 非方形时 PSD 与坐标转置 | v1 要求正方形并交叉校验；非方形 fail fast |
| `optimized/info.json` 成为被消费契约 | 上游改字段会静默破坏 | 单一 parser、v3 PSD/PNG fixtures、完整输入 fingerprint |
| 只看文件存在 resume | 配置或 overrides 改了仍复用旧 rig | stage manifest + 输入/配置/算法/输出摘要 |
| B/C 把同一 `rig.json` 都声明为 output | C 改写后 B 摘要永久失配，昂贵 mesh/weights 每次重算 | stage-owned path 集合两两不交；B 写私有 `RigGeometryCache`，C 独占并一次性写完整 RigDocument |
| `export_manifest.json` 存在即触发 item skip | D/E encoder、validator 或 artifact 已变化仍交付旧 completed 包 | G 成为真实 terminal stage；`skip_completed` 递归验证 A-E/G fingerprints、manifests 和 artifact SHA，陈旧 G 先撤销终态 |
| A/B 失败只留在可删 cache/批次日志 | 无 per-item 可寻址诊断，`diagnostic → overrides` 闭环断裂 | G 独占公共 `rig/error.json`；任一已枚举 item 失败都发布稳定 failure record，成功时与 export manifest 互斥 |
| `tblr_split=false` 或腿未拆分 | 不能安全生成双侧可动骨骼 | 不改默认值；merged/partial 状态 + 明确失败/刚性降级 |
| 背面/侧面角色 | source `-l/-r` 不等于可靠解剖侧 | 内部 `xmin/xmax`；`anatomical_side` 独立可空 |
| Rig schema 演进 | 旧 rig/override 无法读取 | `schema_version` + 集中迁移；首版未发布前不承诺 v0 |
| SDPose 约 0.95B / 官方包数 GB | 启动、显存、依赖冲突、批量吞吐风险 | 独立 worker/extra、默认关闭、Body 17 优先、阶段结束释放 |
| RTMW 包 229 MB | fallback 仍有缓存和 provider 成本 | 独立 ONNX extra、默认关闭、固定 graph 契约 |
| pose 权重和训练数据许可 | 不能稳定分发、镜像或商用 | 每 provider coherence group + SHA-256 + 基础模型/数据许可清单 |
| 静态 PSD 没有新表情纹理 | procedural blink/smile 可能视觉失真 | 默认 core profile 把表情设为 optional；分层 blink validator；严格 avatar profile fail fast |
| 共享 texture pack 超出 4×2048² | 两种格式无法满足冻结的资源预算 | A 阶段精确 dry-run fail fast，C 冻结同一 plan；不降采样、不回退散页；以后通过 profile/version 显式提高预算 |
| D/E 独立 PNG encoder 却要求字节相同 | zlib/filter/chunk/metadata 或依赖升级使同像素输出不同字节 | C 只编码一次 canonical PNG；D/E byte copy，raw/encoded 双 SHA 闭环，encoder 配置只进入 C fingerprint |
| A/B/E 对 texture region 粒度理解不同 | dry-run 与最终 plan 不一致，或稀疏 part 被偷偷重裁 | v1 固定一 canonical part 一 region；所有 component 与未来 skinning ArtMesh 共享它，分量级 packing 留给新 schema |
| Spine major/minor 不匹配 | runtime 直接拒绝数据 | 首版固定 4.2；其他版本明确不支持 |
| optional preset 被误当成必须双格式交集 | Spine 原生能力被无谓删除，或目录扫描得到错误能力结论 | `optional_preset_parity=per_format`；公共 manifest 逐格式状态/artifact 闭环，required 仍严格对称 |
| 骨骼权重不能直接进入 Live2D runtime | 文件能加载但独立关节弯曲缺失 | Live2D v1 不转换多骨 LBS，`wave.*` 仅 Live2D omitted；Spine 正常导出，未来 Live2D 先过 Glue gate |
| 完整 Rig bone 树被复制为 Live2D deformer | 大量无 driver section 增加引用和 consistency 风险 | `Live2DDriverLiveness` 剪枝；静态 rest transform 折叠，emitted/pruned/reparent 表与双向可达性 validator |
| 单一 PPU 公式被套到嵌套 deformer | 子层缩到原点、pivot 漂移或 warp 内 ArtMesh 尺度错误 | 逐节点 `Live2DCoordinatePlan`；E0-core 覆盖 root/warp/rotation frame 与 forward/inverse 0.1 px parity |
| E0 frame 结论没有绑定语义契约 | frame codec/layout/stack 顺序变化后继续复用旧假设 | `live2d_frame_contract_digest` 只钉专用 semantic-kernel 源码、descriptor、vectors 与已实测 Core allowlist；完整 compiler/writer hash 只作 provenance/cache |
| production driver/rank 表被混入 frame attestation | 新增纯内容 driver 也要求重新取得 Core 环境并重签 E0 | attestation 只证明 comparator、parent chain 与 runtime 组合语义；`RigidDriverRegistry` 行内容只进入 motion/export fingerprints |
| rigid-driver rank 跨语义域重复 | preset 演进后两个 required driver 落到同一 bone，默认 profile 才在生产时冲突 | built-in rank 全局唯一且 registry load fail fast；当前 v1 固定为 `100/200/300/400` |
| registry startup 错误被塞进 item diagnostics | 产生虚假 part/joint ID，`continue_on_error` 对程序配置错误继续跑 | packaged registry 在 CI 与 job startup 校验；失败无 item staging/report，binding-plan 冲突才使用 item code |
| 同 bone 多个 rigid parameter 的 deformer 身份/顺序不定 | 两个实现都字节稳定却组合出不同顶点 | instance key=`(bone_id, parameter_id)`；`rotation-stack-v1` 显式 rank 外→内；非交换 synthetic E0 fixture |
| candidate enumerator 与 binding selector 漂移 | 罕见 Rig 在生产批次才以 `missing_export_symbol` 失败 | C 物化 immutable candidate universe，E 只能按 `candidate_id` 选取；generated Rig × 全 registry/profile 属性测试证明 superset |
| 同一 RotationDeformer 的 angle/origin 组合顺序错误 | pivot 平移时顶点走错路径 | 固定 parent-local `T(origin)·R(theta)·S(scale)` 语义；E0-core 同参数同时变化并取 9 点实载 |
| parameter default 插值不等于 rest | 模型加载后未播动作就歪斜且 consistency 仍通过 | default-rest invariant；必要时显式 default keyform，首次 update 后顶点/opacity/draw order 实载 |
| 无 Glue blend band 与刚性边界使用不同插值 | stop 间原理性裂缝/重叠 | 移出正式 v1；E0-S 只测量，未来 required joint bend 必须先验证 Glue 或另立可量化遮盖契约 |
| StretchyStudio runtime 实现缺 rotation hierarchy/expressions | 直接移植会产出“可打开但不满足需求”的包 | 只作 section/warp 参考；E 阶段补 RotationDeformer、parent binding、逐层坐标 plan 与 exp3 |
| MOC3 是未公开二进制格式 | 版本、Core 或 writer 变化会导致 consistency/load 失败 | 固定 V4.00 和参考 commit；parser + Core consistency + 实载三层 release gate |
| Cubism Core/SDK 的分发许可 | CI 或最终产品可能不能合法捆绑 validator/runtime | writer 与官方 validator 解耦；发布前完成许可证审计，不自动打包 Core |
| 多个 non-rigid 参数直接修改同一 ArtMesh | 表情和动作组合出现非预期插值或指数级 keyform | deformer 层级组合不算冲突；只对无法拆分的同一 non-rigid target 报 conflict |
| exp3 在 fade 第一帧被当成 full weight | 正确实现被 E0-core 误判 blocked | 零 fade fixture 仍推进至少一帧并确认 weight=1；blend 公式另做底层 API 单测 |
| baked-position/MOC3 上限被当成质量分 | 默认 core 轻易通过却掩盖坐标或视觉错误 | 标记为 capacity guard；质量只看 runtime parity、插值、landmark/像素，纹理另看 TexturePagePlan |
| 双格式只成功一个 | 下游拿到不完整批次且误判完成 | staging 输出；两边验证通过后最后写共同 export manifest |
| export 名称碰撞/Unicode、按格式独立消解或扁平全局去重 | 串骨、空名称、纹理丢失，或合法跨 section 同名被强制加后缀并破坏默认 region lookup | C 阶段冻结 Rig 级 `GlobalExportSymbolTable` 与 typed `ExportNamespaceKey`；名称只在真实 namespace/scope 内唯一，两 exporter 只取子集并验证引用闭环 |
| 新几何依赖 | 环境变大或 OpenCV 包冲突 | 独立 extra；明确 SciPy/scikit-image；不混装两个 OpenCV wheel |

---

## 非目标

1. **可编辑 Live2D `.cmo3/.can3` 工程**。正式交付已经包含 `.moc3` runtime bundle；
   `.cmo3/.can3` 是另一套未公开工程格式，不是 runtime 文件的同义词。除非下游明确要求在
   Cubism Editor 继续编辑，否则不承担其逆向 writer 或 Editor GUI 自动化。
2. **Live2D physics/pose 与实时面捕**。首版不生成 `.physics3.json`、`.pose3.json`，也不接摄像头
   tracking；固定动作/表情只通过 motion/expression 驱动。
3. **交互式画布编辑器和用户预览**。首版只用可 diff 的 overrides、结构化质量报告和测试夹具；
   不输出 GIF/WebM，也不建设播放器。
4. **任意文本生成动作、动作捕捉和物理模拟**。首版只绑定版本化固定 preset；以后接
   text-to-VRMA/视频姿态时，先转换为同一个 `MotionClip`，不能绕过 RigDocument。
5. **多角色**。输入契约是一张图一个角色；检测到多个主体直接报错。
6. **自动切开粘连的 amodal 双臂/双腿纹理**。首版只接受已有 split 或可靠连通域拆分；
   pose 点不能替代图像分层。
7. **支持任意导出版本**。首版只有 Spine 4.2 与 Live2D MOC3 V4.00；升级任何一边都要新增
   明确 adapter/compatibility gate。
8. **模型量化**。等 SDPose/RTMW 在本项目数据上证明有增益后再单独设计，不能提前优化
   未证实的路径。

---

## 验证与验收

1. **版本/tag 契约**：fixture 的 `layerdiff/manifest.json.tag_version=v3` 能解析；缺失、未知版本、
   把 `head/hairf/hairb` 当 v3 最终 canonical tag 均 fail fast。
2. **双 PartSource**：默认 PSD fixture 与 optimized PNG fixture 生成相同 canonical parts；
   root PNG 即使存在也被拒绝，缺层、重复 tag、RGBA/depth 不配对均失败；PSD fixture 额外包含
   透明边，重载后的 stored layer rectangle 必须与 `xyxy` 精确相等，而 alpha-content bbox 只需
   是其子集。
3. **双 `info.json`**：传入根 Marigold `info.json` 必须得到 `postprocess_info_required`，不能返回
   空 geometry；final part 缺 `xyxy/tag/depth_median` 任一字段均失败。
4. **canvas 契约**：768/1024/1280 fixture 与 `src_img`、manifest、PSD 一致；2048 或其他非白名单
   edge 必须得到 `unsupported_auto_rig_canvas_resolution`，伪造非方形 `frame_size` 必须得到
   `unsupported_non_square_frame`。测试还要断言 `SEE_THROUGH_PROFILES` 的 resolution 唯一集合精确
   等于 auto-rig 白名单 `{768,1024,1280}`；上游新增档位时必须显式适配。不写一个无法证伪的
   通用 H/W 顺序测试。
5. **分辨率门槛**：按 768/1024/1280、极端原图长宽比、小主体和细部宽度分桶，报告 joint
   resolved 率、轮廓误差和刚性降级率；A 阶段必须先定义并达到阈值。
6. **Rig 往返**：内存 Rig → JSON → load 后语义相等，所有 ID、capability、clip、expression、
   `PrimitiveCandidateSet`、`GlobalExportSymbolTable` 和 texture-page 引用有效；交换 exporter 顺序或
   改变 per-format pruning 后，完整 candidate/symbol-table digest 与共同 typed keys 的
   `{namespace_key, export_name}` 不变。另做 registry-driven 属性测试：生成覆盖缺失/merged/split bones、同一 preset 跨 torso/head
   channels 和可选 expression targets 的 Rig 集合，遍历测试运行时发现的全部 profile/preset/parameter/
   primitive registry entries，断言每次 `binding_plan_candidate_ids ⊆ candidate_universe_ids` 且
   `binding_plan_keys ⊆ candidate_universe_keys`。测试不得硬编码当前四个 preset；新增 registry row
   必须自动进入矩阵。故意让 selector 返回未知 candidate 的 mutation fixture 必须在 writer 前失败。
7. **缓存失效与单写者**：修改 manifest、mask、info、payload、tag registry、相关配置、preset 版本或
   overrides，精确失效相应 stage；只改日志级别不失效。完整跑一次后原样重跑，A/B/C 都必须命中
   cache 且所有 output SHA 不变；故意损坏 `rig.json` 的 C-owned 字段时只失效 C→D/E→G，B 的
   `RigGeometryCache`、manifest、mesh/weight SHA 均不变，且测试 spy 证明 B compute/writer 调用次数为
   零；损坏 B cache 才失效 B 及下游。
   另断言所有 stage payload output path 两两不相交、manifest 不自哈希且非终态 manifest 的 SHA 被
   直接下游记录、B
   完成时公共 `rig.json` 尚不存在、部分 RigDocument 被
   D/E 拒绝；把 `rig.json` 同时放进 B/C manifest 的 mutation fixture 必须得到
   `stage_artifact_ownership_conflict`。completed fixture 保留旧 `export_manifest.json` 后修改 D encoder
   fingerprint，`skip_completed` 必须为 false，只重跑 D/G 而复用 E；仅删除 G manifest 时复用 A-E、
   只重跑 G。只有 export manifest、篡改任一 D/E artifact、错误 C/D/E manifest SHA 或旧 validator
   fingerprint 都不得 skip，并必须在执行最早失效 stage 前撤销旧 terminal artifacts。
8. **左右约定**：不对称 fixture 断言 source `-r` 对应较小 x，内部只产生 xmin/xmax；
   `tblr_split=true` 同时覆盖成功替换和保留 base tag 两条路径。
9. **几何关节**：覆盖直肢、弯肢、袖口毛刺、断裂 mask、交叉/粘连，并逐项覆盖
   head_base/head_top/wrist/hand_tip/ankle/toe observation；resolved 关节在合法 mask 区域，错误
   场景产生 unresolved 而非假坐标。geometry-only wrist false-resolve `≤5%` 且不设 recall 下限；
   eligible ankle conditional resolved `≥80%`、false-resolve `≤5%`。
10. **网格/权重**：不跨连通分量，triangle 索引合法且面积大于零；每顶点 1-4 个有效
    influence 且和为 1；分辨率缩放后弧长权重分布近似不变。
11. **纹理页计划**：A 对每 canonical part 的真实 padded bbox 做 dry-run，C 用同一 packer/排序
    重算出的 plan 必须逐字段一致；B 的连通分量以及 E0-S/未来 Glue 的 skinning ArtMesh 拆分不得改变 region
    数量/rect。断言 2048²、最多四页、padding/extrude/no-rotation、无 region overlap、UV/rect
    闭环、`rgba_sha256` 与 `encoded_png_sha256` 各自正确。测试 spy 必须证明 C encoder 每个 used page
    恰好调用一次、D/E image encoder 调用次数为零；两 exporter 只做 byte copy，三份对应文件 SHA
    逐页完全相同。canonical PNG 不含非确定 metadata，encoder version/settings 改变必须失效 C 及下游；
    分别核对
    `budget_occupancy=sum_area/(4×2048²)` 与
    `used_page_fill=sum_area/(used_pages×2048²)`。超预算必须在 A 返回
    `texture_budget_exceeded`，不能等 C、不能散页或静默缩图；真实样本按 resolution/tblr_split
    统计两个比率、page count 和 MaxRects 成功率，不能用总面积估算替代实际装箱。
12. **动作/表情**：preset 同输入逐字节确定；默认 core profile 只要求四个不依赖新表情纹理的
    结构/骨骼动作；`head_nod` 与 `head_shake` 必须各自产生非空、可区分的 landmark 轨迹，并明确
    标记为 2D stylized motion，不能以标准参数 ID 冒充 3D 新视角；merged
    limb 不产生 wave；完整 limb 的正式 dual-runtime fixture 必须在 Spine 生成 wave、在 Live2D 以
    `live2d_joint_bend_requires_glue` 省略，并与 manifest 的 `supported_formats=["spine_4_2"]`、artifact
    引用逐项一致；layered blink 分别验证眼白/虹膜残留和睫毛厚度；strict avatar 缺 required
    capability 时失败。两个 exporter 消费同一份 preset 语义和逐格式决策，不能各自猜测。
13. **Spine 结构**：weighted/unweighted/animated golden fixtures 断言 flat arrays、bone indices、
    multi-page/multi-region atlas、effective attachment paths、namespaced global symbol table 子集/摘要、
    坐标转换和 animation timeline。同一 `topwear` 派生出的 Spine slot/attachment-key/atlas-region 与
    Live2D Part/ArtMesh 允许保持同名；两个 skin/slot scope 内的 attachment key 也允许同名，但 actual
    attachment name 仍在自己的 skeleton-global namespace 验证，必要时显式写不同 actual name/path；同一
    namespace 内两个不同 typed key 同名则必须 `export_name_collision`。省略 `path` 和显式 `path`
    两条 fixture 都要恰好命中一个 region；exporter 内故意再次 sanitize、改变 namespace 或按存活集合
    消解名称的 fixture 必须失败。
14. **Spine 实载**：opt-in 4.2 Editor/runtime 加载并逐个播放 required clips，不能出现 recovery、
    missing region、schema error 或 attachment 跳变；发布 Spine 支持前必须真实执行一次。
15. **Live2D 普通 CI**：不依赖 Cubism Core 的 golden fixtures 断言 MOC3 header version 3、SOT
    offset/count、Rotation/WarpDeformer、parent_deformer_indices、ArtMesh index/UV/texture、parameter
    binding、rest/deformed keyforms、schema 和 deterministic bytes；model3 的
    Moc/Textures/Motions/Expressions/Groups 全部可解析。每个声明可见效果的标准/custom parameter
    都必须绑定非空 RotationDeformer、WarpDeformer、ArtMesh keyform 或 opacity target，否则得到
    `live2d_parameter_binding_missing`。fixture 还要解析 `Live2DCoordinatePlan`，证明每个非 root 节点
    都有明确 parent frame、forward/inverse 版本和输入摘要，不允许出现“全节点套 root PPU”路径；
    Parts/Deformers/ArtMeshes/Parameters IDs 必须全部来自同一 `GlobalExportSymbolTable` 的对应
    namespace，并与 model3/cdi3、binding、motion/expression、artifact fragment 的 typed key 及跨格式
    base 逐项闭环；Part 与 ArtMesh 使用相同裸 `topwear` ID 的正例必须通过，同一 ID array 内重复则
    `export_name_collision`。缺 symbol、同 namespace 碰撞、非 ASCII、`≥64` bytes 或按剪枝子集重新
    命名必须失败。反向遍历证明每个
    emitted `(bone_id, parameter_id)` deformer instance 有直接 driver，故意加入的 forearm/shin 静态枝
    必须被剪除；
    `live parent → static intermediate → driven child` fixture 必须折叠中间 rest frame、保留 child driver，
    否则得到 `live2d_dead_deformer`。剪枝前后的 default rest vertices 必须一致。
16. **Live2D deformer、坐标与 non-rigid 逼近**：E0-core 的
    `root→warp→rotation→rotation→ArtMesh` 在 rest、单层和同时驱动时，每个参数区间 9 点的 runtime
    顶点都与解析 frame stack 相差 `≤0.1 px`；全部 vertex/control point/origin 的 coordinate
    round-trip 同样 `≤0.1 px`，超阈值返回 `live2d_coordinate_roundtrip_failed`。同一 RotationDeformer
    的 angle 与 origin 必须在一个参数上同时变化。所有参数设为 default、完成首次 model update 且
    尚未应用 motion/expression 时，vertices/rest transforms 必须 `≤0.1 px`，opacity/draw order 也与
    rest 相等；缺失显式 default key 导致插值偏离时返回 `live2d_default_pose_mismatch`。随后验证
    `T(origin)·R(theta)·S(scale)` 顺序。`RigidDriverRegistry` load test 必须证明 built-in rank 全局
    唯一，当前集合精确为 `{100,200,300,400}`；把 `idle` 与 `head_nod` 同时绑定到 head 的 fixture
    不得产生 rank conflict。复制任一 built-in rank 的 registry mutation 必须使 CI 失败；worker
    defense-in-depth 路径必须在 item 枚举前返回 `invalid_rigid_driver_registry`，且没有
    `rig/report.json`。同一 bone 的两个 synthetic parameter 使用不同 pivot/origin，正反组合
    的 fixture 差必须 `>0.1 px`；emitted parent indices、runtime vertices 和 report 只能匹配
    `rotation-stack-v1` 的 lower→outer 语义，不能以任一稳定字典序通过。另做已枚举 item 的负例：
    同一 `(bone_id, parameter_id)` instance 在两个 binding 中携带不同 rank，必须得到 item-level
    `live2d_deformer_order_conflict`。另断言只改 production registry rows 时 frame-contract digest 不变，而
    motion/export fingerprints 必变。RotationDeformer 标量不计入 17-stop/baked-position
    预算。required breath 与其他 baked non-rigid channel 验证逐顶点 `E_i ≤ 0.05 × r_i`、渐细端预算、
    17-stop、100 万 positions 和 64 MiB 的失败路径；三个数字只断言 capacity guard，不生成质量分。
    E0-S 则必须观测并记录无 Glue 边界的弦割/seam，产生 experimental warning 而不是要求零缝或
    阻塞 E0-core。默认 core 四个 required preset 同时存在不得触发 `live2d_parameter_conflict`。
17. **Live2D expression 行为**：E0-core 在同一个有可见 keyform binding、不会 clamp 的参数上先播放
    motion；底层 API 在 full weight 下断言 Add/Multiply/Overwrite 为 `p+v`、`p*v`、`v`。端到端
    exp3 固定零 fade、推进至少一次 update 并确认 effective weight=1 后，才断言参数与
    landmark/像素变化；禁止用 manager 第一帧冒充 full weight。任一 blend 只改数值不改渲染、
    或更新顺序不符，均报 `live2d_expression_application_failed`。
18. **Live2D startup、attestation 与 opt-in release gate**：structural/release tier 都先验证 packaged
    registries，再在枚举 item 前重算 `live2d_frame_contract_digest` 并运行 attested pure vectors；
    registry 失败使用 job-level `invalid_rigid_driver_registry`；缺文件或 frame/layout semantic-kernel、
    codec/transform/rotation-stack descriptor/vector 任一不匹配时返回
    `live2d_coordinate_schema_unverified`，
    不创建 staging。完整 compiler/writer provenance 改变只失效 export cache 并进入报告，不单独使
    coordinate gate 失败。普通 CI 消费现有 attestation、跑完整 structural pipeline，但不重跑 E0、
    不需要 Core，也不能写 completed manifest。release tier 在枚举 item 前还要求 Core binary 位于
    attested allowlist，且 SDK harness 的 validator protocol digest 一致；缺失得到
    `live2d_release_gate_unavailable`，未 attested 得到
    `live2d_core_unattested`，不得由 worker 临时生成 attestation。startup 通过后，对每个正式输出执行
    `csmHasMocConsistency`，再用官方 SDK 或 Viewer 加载、渲染非空默认帧，驱动参数 extrema，并
    逐个播放 required motion/expression；另用最小 namespace fixture 让 Part 与 ArtMesh 使用相同裸 ID，
    证明官方 runtime 能分别索引两个 section。若实载不支持，必须升级 `namespace_schema_version` 并合并
    相应 collision scope，不能按 item 临时加后缀。该结论不属于 frame-contract attestation；
    attestation 也不能替代逐 item release validation。
19. **双格式原子交付与状态**：对同一 Rig，Spine 与 Live2D 的 required preset ID、duration、
    loop 和 capability 结论必须一致，语义 landmark 在容差内；optional preset 允许格式不对称，
    但每个实际 animation/motion/expression 必须与 `motion_manifest.formats` 双向闭环，禁止未声明文件
    或声明 supported 却缺 artifact；`export_manifest.json` 中 C/D/E manifest、三组 artifact-set、
    motion manifest、global symbol table、profile 和 validator fingerprints 必须逐项匹配当前磁盘与
    G expected fingerprint。
    无降级时写 `completed`；正式双格式
    `--allow-partial` 仅在两边验证通过且 required capability 仍满足时写
    `completed_with_degradation`。`spine_4_2_dev --allow-partial` 只写
    `stage_validated_with_degradation`。任一 A-E item failure 都不写 `export_manifest.json`，由 G 原子写
    公开 `error.json` 并返回对应错误；成功时 G 删除 error 后写 export manifest。两者 XOR，G manifest
    的 output SHA 必须匹配当前终态文件。A/B failure fixture 在删除全部 cache 后仍能从 item 根目录读取
    error 和 repair；D/E 并行同时失败时，`failed_stages/failure_records` 必须按 stage name 稳定聚合两条，
    不能“最后一个异常覆盖前一个”；job-startup failure 则不得伪造 per-item error。D 的 stage report
    可以保留，但不等于正式交付完成。
20. **SDPose 契约**：真实 Body 模型组件摘要、`1024×768` 预处理、17 点/score 输出、timestep
    和坐标反变换固定测试；与官方脚本对同一 crop 做 parity 对比。
21. **RTMW 备选契约**：真实 ONNX graph metadata、SHA-256、预处理和输出 shape 固定测试；SDK
    `pipeline.json` 的错误 image_size 不得污染运行时。
22. **姿态价值评测**：建立人工标注动漫子集，按普通、弯肢、交叉、缺失分层，对比 geometry、
    geometry+SDPose、geometry+RTMW 的 normalized joint error/PCK、左右交换率、unresolved 率、
    吞吐/峰值显存和人工 override 数。仅“落在 mask 内”不足以证明正确；没有显著降低 override
    数时，两个模型都保持实验性并默认关闭。

阶段验收：A 先证明输入和几何质量；B 再做 mesh/weights；C 冻结 common preset、profile、
GlobalExportSymbolTable、TexturePagePlan 与 canonical PNG bytes；
D 通过 Spine setup/animation 实载；E 通过 Live2D consistency、渲染、motion/expression 实载，
G 再验证完整 DAG 并发布成功/失败终态；F 只有在评测结果支持时才进入可选产品路径。

---

## 审查修订对照

| 审查发现 | Revision 2/3/4/5/6/7/8/9/10/11/12/13 处理 |
|---|---|
| R1 把 canvas 错写成原图像素 | 冻结为 LayerDiff 方形 letterbox canvas；明确无原图反变换契约 |
| R2 漏掉 tag 版本与 v3 实际分支 | 更正“完全未持久化”的判断：从 `layerdiff/manifest.json` 强制读取；冻结 v3 tag 流转和 dead branch |
| R3 根/optimized 两个 `info.json` | 禁止根 schema，逐 part 强制 `tag/xyxy/depth_median` |
| R4 根 PNG 被误当 PartSource | 更正理由并无条件禁用；只接受 final PSD 或 optimized PNG |
| R5 方形掩盖 PSD H/W 错位 | 本地实测 psd-tools 参数顺序；v1 非方形 fail fast，不伪造通用顺序契约 |
| R6 原 C 阶段 MaxRects 过重 | 后移只对 Spine 单格式成立；恢复共享 packer 不是因为 Live2D 有纹理页格式硬上限，而是双 exporter 的 region/page 必须共享。Revision 6 把同算法 dry-run 前移到 A；Revision 13 进一步由 C 单次编码 canonical PNG，D/E 只复制字节 |
| R7 pose 候选过时 | SDPose-OOD Body 升为首选评测候选，RTMW-l 降为已核实 ONNX 备选 |
| 批量交付不需要预览/GIF/WebM | 删除公共 preview 产物，加入 capability、common preset profile 和双格式 animations/motions/expressions |
| 默认 `save_to_psd=true` 不产 optimized 最终逐层 PNG | 改为 final PSD / optimized PNG 双 PartSource 契约 |
| `rig.json` 与阶段 JSON 可能形成多个事实源 | `rig.json` 唯一公共事实源；Revision 12 进一步固定 C 为唯一 writer，A/B cache 使用独立 schema 且 exporter 禁读 |
| 文件存在式 resume 会复用陈旧结果 | 引入 stage manifest、输入/配置/算法/产物摘要 |
| confidence 跨来源不可直接相加 | 保存 observations，使用约束决策表，不平均未校准分数 |
| 上游深度聚类没有处理四肢 | 修正 F10；merged limb 明确失败/降级 |
| 骨骼只有 pivot、权重只有一个标量 | bone head/tail + 通用 per-vertex influences |
| Spine 不只丢权重，mesh schema 也错误 | 独立 4.2 encoder、atlas、symbol table 和三层验证 |
| Live2D 预览/导出各自绑一遍 | 两个 exporter 都只读 RigDocument；Live2D compiler 不得重新分析图层或绑骨 |
| 正式交付必须含 Spine 4.2 与 Live2D | Revision 4 把 Spine 4.2 + MOC3 V4.00 runtime 设为原子双输出；editable CMO3 仍排除 |
| StretchyStudio runtime 能打开但缺关键能力 | 不把参考 writer 当成完成；其 runtime builder 实际只发射 root WarpDeformer，E 阶段另补 RotationDeformer hierarchy、parent binding、逐层坐标 plan、exp3 与官方 runtime 行为验收 |
| 刚体骨骼被错误烘焙为 ArtMesh 顶点 | Rig bone 树改映射为嵌套 RotationDeformer；统一缩放/旋转/中心平移在 runtime 组合，非均匀形变使用 WarpDeformer |
| 非刚性 keyform 没有误差/体积契约 | 对 warp、blink/talk 和实验 joint blend 使用 adaptive stops、逐顶点 5% 局部半径、17 stops、100 万 baked positions、64 MiB 和禁用多参数 grid；三个上限只作 capacity guard |
| required motion 在同一 head/torso ArtMesh 冲突 | 刚体参数拆成可组合的嵌套 deformer；`live2d_parameter_conflict` 只约束无法拆分的直接非刚性多 driver |
| TexturePagePlan 粒度未冻结 | v1 固定按 canonical part 的完整 `xyxy` 装箱；分量和未来 skinning 子网格共享 part region，因此 A dry-run 与 C plan 可逐字段一致 |
| expression 第一帧并非 full weight | 公式由底层 API 在 full weight 单测；端到端 fixture 固定零 fade、推进至少一次 update 后再断言参数和像素 |
| atlas 面积率分母含糊 | 拆为固定四页预算占用率 `budget_occupancy` 与实际已用页装填率 `used_page_fill` |
| 无 Glue blend band 的误差门与零缝门互斥 | 从正式 Live2D v1 移除 joint-bend compiler；Spine 可保留 wave、Live2D omitted，E0-S 只量化 seam，未来 required path 先验证 Glue |
| 嵌套 deformer 被错误地共用 root 坐标公式 | 新增 `Live2DCoordinatePlan` 与 frame stack；E0-core 实载 root/warp/rotation/ArtMesh 的 forward/inverse 和 0.1 px round-trip |
| 同一 rotation 节点 angle/origin 同时变化未测 | 固定 `T(origin)·R(theta)·S(scale)`，E0-core 在同一参数区间取 9 点验证 |
| MOC3/baked 上限被误当质量指标 | 保留为 capacity guard；质量另看 runtime transform、插值、landmark/像素，纹理容量由 TexturePagePlan 负责 |
| optional `wave.*` 在两格式间出现能力分叉 | 不再伪装成双格式交集；正式 profile 固定 `optional_preset_parity=per_format`，公共 motion manifest 逐格式记录状态、原因和 artifact，required preset 仍严格对称 |
| Live2D v1 会生成无 driver 的远端肢体 deformer | 先冻结 `Live2DBindingPlan`，再用 `Live2DDriverLiveness` 按计划可达性剪枝并折叠静态 rest transform；禁止从 writer 输出反推活性、手写 anatomy 白名单或保留 pass-through dead node |
| 参数 default 可能插值成非 rest 初始姿态 | 增加 `default_rest_invariant`；E0-core 与逐模型 validator 在首次 update 后比较 vertices、opacity、draw order 和 rest transform，必要时强制显式 default keyform |
| coordinate schema 错被当成逐 item 运行期诊断 | E0-core 产出 digest-pinned `attestations/live2d-frames-v1.json`；正式 worker 枚举 item 前一次性校验摘要，item 内只报告自身 round-trip/default-pose 错误 |
| deformer 身份在 bone 与 `(bone, parameter)` 间含糊 | v1 固定一 rigid parameter 一个 RotationDeformer instance；typed pair key + 显式 stack rank，E0 用非交换变换证明顺序 |
| 完整 compiler/writer hash 使 frame attestation 过度失效 | gate 缩为 `live2d_frame_contract_digest`，只 hash 隔离的 semantic kernels/descriptor；完整源码仅 provenance/cache，普通 CI 消费现有证明并跑 structural vectors，不负责重签 |
| Live2D 没有共享 export-name 契约 | C 阶段从完整 symbol universe 生成 Rig 级全局表；Spine/MOC3/JSON/report 只取子集，剪枝不得改变名称 |
| rigid-driver rank 重复依赖当前 preset 恰好分骨 | `RigidDriverRegistry v1` 的 built-in rank 改为全局唯一；registry load 与 CI 都拒绝重复值，移除 torso/head 永不共存的隐式假设 |
| production rank 表进入 Core-dependent attestation | `rotation-stack-v1` 只证明排序和组合语义；production rows 排除出 frame digest，改由 motion/export fingerprint 精确失效 |
| candidate universe 的超集关系只在运行期碰撞 | C 物化同一 enumerator 的 immutable typed candidates，E 禁止自行构造 key；generated Rig × 全动态 registry/profile 属性测试在 CI 证明两个 subset 不变量 |
| B/C 都改写 `rig.json` 导致 resume 永久失效 | B 改为只写私有 `RigGeometryCache v1`；C 独占并原子写完整 `RigDocument v1`，stage output paths 两两不交，增加精确 resume 边界 fixture |
| 整个 symbol universe 强制裸名称全局唯一 | 改为 canonical typed `ExportNamespaceKey`；同名只在真实 format/section/skin-slot scope 内碰撞，typed key 映射和跨格式 base 仍稳定，Spine effective region path 单独闭环 |
| built-in rank 重复使用 item 诊断码 | 拆分 job-startup 与 item diagnostics；registry 错误在 CI/启动期失败且无 item report，`live2d_deformer_order_conflict` 仅表示已枚举 item 内 binding 自相矛盾 |
| `export_manifest.json` 由不存在的 join 拥有且可能按存在性 skip | 新增具名 G terminal stage/manifest；成功描述符绑定 C/D/E manifest 与 artifact sets，auto-rig skip 必须验证完整 DAG，不能复用 see-through 的文件存在 predicate |
| A/B failure 没有公共 per-item artifact | 不让失败 stage 动态共写同一路径；G 统一把规范化 failure record 发布为 `rig/error.json`，cache 删除后仍可用于 overrides，成功时与 export manifest 互斥 |
| D/E 分别编码 PNG 无法保证相同字节 | C 在公共 shared path 每页编码一次并记录 raw/encoded SHA；D/E encoder 被禁止，只能 byte copy，增加调用次数和 metadata regression fixture |
| 默认 profile 会因静态表情素材失败 | 默认改为 dual core，只要求不依赖新表情纹理的结构/骨骼动作；表情按 capability，strict avatar 变为 opt-in |
| head/wrist/ankle 等 joint 来源不明 | A 阶段新增逐 joint observation 与 eligibility；wrist 允许系统性 unresolved |
| PSD bbox 可能被 alpha 裁小 | psd-tools 1.17.4 与项目 save_psd 透明边实测均保持 stored rectangle；保留精确断言 |
| RTMW dims、体积仍靠猜 | 下载官方包并记录真实 graph、大小、SHA-256；发现辅助 JSON 不一致 |
| RTMW 下载归错 owner | 改用 shared ONNX resolver + coherence-group inventory |
| Apache 代码许可被等同于模型权重许可 | 增加 Cocktail14 数据集许可证据 gate |
| 姿态模型评测只看关节是否在 mask 内 | 增加人工 GT、PCK/归一化误差/override 数 |
| SciPy/skeletonize 被误称为现有依赖 | 拆出明确的 auto-rig extras，使用成熟 skeletonize 实现 |
| “现有代码改动为零”不真实 | 列出 config/GUI/packaging/inventory/tests 触点并分阶段提交 |

---

## 待确认

- SDPose-OOD Body 完整组件的固定 revision/SHA、Windows 环境兼容性、峰值显存和批量吞吐；
  本轮不做量化，也不把 ComfyUI 重打包权重当作官方 Body 等价物。
- SDPose 与 RTMW 应分别镜像到哪个 canonical Hugging Face/ModelScope coherence group；本地文件
  模式可先实现，但正式 GUI 下载不能绕过统一模型源设计。
- Stable Diffusion v2 基础权重、SDPose 训练数据及 Cocktail14 各数据集对权重再分发、商业使用
  和镜像的具体约束。
- 真实 see-through 数据上 medial-axis 主路径的 resolved/unresolved 比例，尤其是宽袖、裙装、
  交叉肢体和只输出 merged `legwear` 的样本。
- 768/1024/1280 letterbox canvas 在极端长宽比、小主体样本上的细部几何通过率。
- 在代表性真实 see-through 批次上，按 768/1024/1280 与 `tblr_split` 分桶统计 padded bbox
  面积率、part 数、MaxRects page count 的 p50/p95/max、`texture_budget_exceeded` 比例和装箱成功率；
  这是 TexturePagePlan 的可行性 gate，必须先于峰值内存/加载性能评测，并在实施计划前冻结可接受
  的批量失败率。
- 单帧静态纹理生成 `blink/talk/happy/sad/surprised` procedural deform 的可接受率；不达标时
  `dual_runtime_avatar_v1` 应严格失败，而不是降低质量门槛。
- 可用于 release gate 的 Spine Editor 4.2 或匹配 runtime 环境及许可证安排。
- 可用于自动 release gate 的 Cubism Core/SDK 版本、`csmHasMocConsistency` 调用方式和许可证/
  再分发安排；普通安装不应捆绑来源不明的 Core DLL。
- MOC3 V4.00 中 root model、warp-local、rotation-local 与 ArtMesh keyform 的确切单位和变换顺序。
  `WARP_DEFORMERS.md` 是 CMO3 Editor/Hiyori 的逆向证据，现有 StretchyStudio runtime writer 又只
  覆盖 root warp；E0-core 必须实载 frame stack 后才能冻结 `live2d-frames-v1`，不能从数值范围猜。
- RotationDeformer 七类 keyform section、`base_angle` 换算、局部 origin、
  `parent_deformer_indices`、ArtMesh parent binding 与官方 runtime 的 parity。E0-core 必须同时覆盖
  两个不同参数的嵌套节点、同一节点 angle/origin 同时变化，以及同 bone 不同 pivot 的两个
  `(bone, parameter)` instances 按 `rotation-stack-v1` 组合，不能只看 consistency 返回值。
- parameter default 到 keyform binding 的实际插值语义，以及只写 min/max 时能否在非中点 default
  精确还原 rest。现阶段不假设一定需要第三个 key；E0-core 必须用正/负 fixture 签署
  `default_rest_invariant`，不一致时 compiler 必须补显式 default keyform。
- `live2d_frame_contract_digest` canonical descriptor 的最终序列化格式、纯函数向量集、专用
  frame/layout semantic-kernel 模块边界和相关 MOC3 layout descriptor 的最小覆盖范围；它不能重新
  膨胀成完整 compiler/writer hash，也不能把实际语义逻辑藏在未入 digest 的 orchestration 层或漏掉
  `rotation-stack-v1` 的 comparator/parent-chaining 语义；production `RigidDriverRegistry` rows 则必须
  排除并进入 motion/export fingerprints。`attestations/live2d-frames-v1.json` 必须由固定 E0-core 环境生成，不能由正式
  worker 临时生成。v1 的 Core compatibility 使用逐 binary 实测 allowlist，并冻结
  `e0_validator_protocol_digest`；若要扩展，必须另做多版本 E0 矩阵，不能只写未经验证的 semver
  range 或把完整 harness 源码 hash 重新塞回 coordinate gate。
- Glue、GlueInfo、GlueKeyform section 的 Python writer 与官方 runtime parity。它不属于正式 v1；
  只有在未来 profile 要求独立肘/膝弯曲时才先做单独 E0 gate。E0-S 的无 Glue seam 数据只能用于
  评估，不能作为可交付 skinning 的替代证据，也不得退回整 part 顶点烘焙。
- 固定 4×2048² TexturePagePlan 在目标 Native/Web runtime 的峰值纹理内存、加载时间和画质；
  预算已经是正式前置，实测只决定是否需要新增更严格的部署 profile，不能回退到散页。

---

## 参考资料

- [MMPose Cocktail14 / RTMW model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html)
- [OpenMMLab RTMW-l 384×288 ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip)
- [DWPose 官方仓库与指标](https://github.com/IDEA-Research/DWPose)
- [See-Through v0.0.2 UNet 配置（`tag_version=v3`）](https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d/commit/a369334913e2f04ec3a6a0781b91c8131fe84706)
- [SDPose-OOD 官方仓库](https://github.com/T-S-Liang/SDPose-OOD)
- [SDPose 官方项目页与 OOD 指标](https://tsliang.top/SDPose/)
- [SDPose-OOD Body 官方模型仓库](https://huggingface.co/teemosliang/SDPose-Body)
- [Spine JSON export format](https://esotericsoftware.com/spine-json-format)
- [Spine atlas format（支持多 page）](https://esotericsoftware.com/spine-atlas-format)
- [Spine version compatibility](https://esotericsoftware.com/spine-versioning)
- [StretchyStudio Live2D exporter（固定 commit）](https://github.com/MangoLion/stretchystudio/tree/24a83a27ba43e43e9d2e3de5e33994594e6199c2/src/io/live2d)
- [StretchyStudio MOC3 格式与验证记录](https://github.com/MangoLion/stretchystudio/blob/24a83a27ba43e43e9d2e3de5e33994594e6199c2/docs/live2d-export/MOC3_FORMAT.md)
- [StretchyStudio WarpDeformer 层级与局部坐标记录](https://github.com/MangoLion/stretchystudio/blob/24a83a27ba43e43e9d2e3de5e33994594e6199c2/docs/live2d-export/WARP_DEFORMERS.md)
- [StretchyStudio CMO3 deformer emitter](https://github.com/MangoLion/stretchystudio/blob/24a83a27ba43e43e9d2e3de5e33994594e6199c2/src/io/live2d/cmo3/deformerEmit.js)
- [Live2D embedded model data / MOC3 export](https://docs.live2d.com/en/cubism-editor-manual/export-moc3-motion3-files/)
- [Live2D Cubism 4.2 deformer 与线性插值收缩](https://docs.live2d.com/4.2/en/cubism-editor-manual/deformer/)
- [Live2D Cubism 4.2 RotationDeformer](https://docs.live2d.com/4.2/en/cubism-editor-manual/making-and-rotation-of-rotationdeformer/)
- [Live2D Cubism 4.2 deformer 父子层级](https://docs.live2d.com/4.2/en/cubism-editor-manual/system-of-parent-child-relation/)
- [Live2D Cubism 4.2 deformer 组合](https://docs.live2d.com/4.2/en/cubism-editor-manual/combintion-of-parent-child-relation/)
- [Live2D Cubism 4.2 parameter keyform 与 mapped-parameter 验证](https://docs.live2d.com/4.2/en/cubism-editor-manual/edit-parameters/)
- [Live2D 多参数 keyform 组合示例](https://docs.live2d.com/en/cubism-editor-manual/multi-key/)
- [Live2D ID 转换与引用风险](https://docs.live2d.com/4.2/en/cubism-editor-manual/convert-id/)
- [Live2D Cubism 4.2 Skinning](https://docs.live2d.com/4.2/en/cubism-editor-manual/skinning/)
- [Live2D model3/motion3/exp3 JSON API](https://docs.live2d.com/en/cubism-sdk-manual/json-unity/)
- [Live2D motion playback](https://docs.live2d.com/en/cubism-sdk-manual/motion/)
- [Live2D parameter operation 与 motion/expression 更新顺序](https://docs.live2d.com/4.2/en/cubism-sdk-manual/parameters/)
- [Live2D expression runtime 与 blend mode](https://docs.live2d.com/en/cubism-sdk-manual/expression-unity/)
- [Live2D expression setup/export](https://docs.live2d.com/en/cubism-editor-manual/setting-and-exporting-facial-expressions/)
- [Live2D Cubism Core Native API reference（parameter default 与 model update）](https://docs.live2d.com/wp-content/uploads/2021/03/NativeCoreAPIReference_en_r8.pdf)
- [Live2D Cubism 4.2 compatibility](https://docs.live2d.com/4.2/en/cubism-sdk-manual/compatibility-with-cubism-4-2/)
- [Live2D texture atlas editor](https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/)
- [Cubism Web sample 动态加载 `getTextureCount()`](https://github.com/Live2D/CubismWebSamples/blob/b1de66b0b1f1cb881d95fb6158622aeb6a2827bd/Samples/TypeScript/Demo/src/lappmodel.ts#L532-L565)
- [Live2D Cubism External Application Integration](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration/)
