# See-Through 图层自动绑骨设计（Auto-Rig）

## Status

**Revision 26，Stage C exporter-neutral 事实链已冻结并实现：版本化 control/preset registry、能力与 binding 决议、逐格式 feasibility、完整 primitive/symbol universe、共享 canonical texture pages、引用闭合的 `RigDocument v1` 及只读 projections 现在由 C 单次组装和事务发布。A/B 私有缓存保持不可变，C 是公共 `rig.json`、`report.json`、motion/expression manifest、全局符号表和 canonical page PNG 的唯一 writer；D/E 只能消费这些事实，尚未在本 revision 宣称 Spine 4.2 或 Live2D runtime exporter 完成。Live2D frame contract 继续使用 Revision 22 由官方 Cubism SDK for Native 5-r.5、Core 06.00.0001 和 D3D11 WARP 签署的结果。NativeVariant 契约保持 Revision 20。正式交付仍为 Spine 4.2 + Live2D runtime；SDK/Core 不随 Python 包分发，组织是否需要另行取得 Release License 仍是发布前外部合规门。**

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
子集，不能各自维护“已占用名称”或临时调用 `sanitize_name()`。空/非法 internal slug 直接失败；preferred
name 超长、碰撞或碰到保留的 Live2D standard parameter ID 时，只在相同 canonical
`ExportNamespaceKey` 内按 `ExportNameCodec v1` 为碰撞类非保留项附加稳定 typed-key suffix；
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

### Revision 14 阶段边界与派生产物完整性审查

| 审查项 | 核对结论 | Revision 14 处理 |
|---|---|---|
| W1 A 执行 MaxRects dry-run，但 packer 版本只进入 C fingerprint | **成立，会把本应精确失效 A 的算法变化拖到 C 才以 plan mismatch 暴露** | `TexturePagePlan` schema、资源 profile、rectangle builder、MaxRects 实现/版本/排序同时进入 A 与 C fingerprint；PNG encoder/Pillow/zlib 仍只进入 C。修改 packer 必须重跑 A，修改 encoder 只从 C 开始 |
| W2 “G 对每个 item 恰好执行一次”与 dev/structural 成功永不 completed 互斥 | **成立。非发布成功既不能写正式 manifest，也不应伪装成 failure** | G 的 success path 只属于正式 release profile；所有已枚举 item 的 A-E production-stage failure 仍统一走 G。dev/structural success 以最后一个活动 stage report 结束，既无 G manifest 也无公共 terminal artifact；`skip_completed` 对这类 job 非法，只能使用 stage resume |
| W3 page/motion/expression 是可变集合，却没有清理旧产物的精确 inventory | **成立。manifest 只核对列出的文件会让 4→1 页或 supported→omitted 后的旧文件继续留在发布目录** | `output_file_sha256[]` 改为 owner stage 的精确集合并增加 inventory digest；重跑先使旧 commit marker 失效，发布时删除该 owner namespace 中不在新 inventory 的旧文件，最后才写 manifest。C/D/E/G validator 拒绝任何未列出的 owner-owned public artifact |
| W4 `rig.json` 被称为唯一事实源，但 D/E 又直接从 `motion_manifest.json` 取 capability 决策 | **成立，是双事实源** | clips/expressions/逐格式决策只以 `RigDocument` 为准；`motion_manifest.json` 改为 C 的确定性只读 projection，绑定 `rig_json_sha256`、语义 section digest 与 symbol-table digest。D/E 从 Rig 构造 binding plan，只交叉验证 projection，不能从 projection 覆盖 Rig |
| W5 C 已冻结 supported/omitted，E 却可因 keyform 误差/预算临时 omit optional preset | **成立，两个阶段同时拥有 capability 决策** | C 在提交 Rig 前运行纯函数 `FormatCapabilityPreflight`，把逐格式 status/reason、planner version/input/output digest 写入 clip/expression decision；D/E 复算同一 planner 并要求摘要相等。所有 omission 只能发生在 C，writer/validator 阶段不再降级；late mismatch 或格式失败使 item 失败 |
| W6 exporter-neutral expression 允许 Add/Multiply/Overwrite，但 Spine 没有 exp3 等价的资产级参数混合字段 | **成立，strict avatar 会得到两份结构合法但组合语义不同的表情** | v1 双格式 expression 只允许 full-weight `overwrite`，并冻结“base motion 后应用 expression”顺序；Spine 产独立 expression animation + manifest runtime application contract，Live2D 写 Overwrite exp3。Add/Multiply 仅作为 E0 的 Live2D writer conformance，不声明 dual-runtime capability |
| W7 canonical PNG 相同被误写成最终 UV 也可共用 | **成立。共享像素/region 不等于 Spine JSON 与 Cubism Core 使用相同 V 轴和 atlas 变换** | C 只冻结 top-left image-space rect 与 canonical UV；D/E 分别用版本化 `FormatUvAdapter` 转换目标 UV，不改像素。Spine golden/runtime 与 Live2D E0 都用四角异色、非对称 mesh fixture 验证实际采样，禁止凭数值范围猜 `v` 是否翻转 |
| W8 shared PNG 没有 alpha-mode contract | **成立。Spine atlas 有显式 `pma`，Cubism renderer/texture upload 也必须与像素是否预乘一致；字节相同仍可在透明边一边发黑、一边发光** | canonical page 固定 straight-alpha sRGB RGBA；Spine atlas 明写 `pma:false`。Live2D report/manifest 声明 target-specific loader contract，Native/Web release harness 按官方 straight→runtime 路径加载；半透明异色 fixture 检测 double/missing premultiply |
| W9 `MotionClip` 没有 interpolation 契约 | **成立。两格式可有相同 key/duration/loop，却在 key 间走不同轨迹** | `MotionClip v1` 固定 30 Hz rational-time 采样和显式 piecewise-linear interpolation；loop 首尾值闭合。Spine 写 linear timeline，Live2D 写 linear segment，运行时按统一时间网格比较。Cubic/Bezier 留给升级后的 motion schema |
| W10 depth 被说成用于 draw order，但没有 canonical 排序/tie-break | **成立。同一份 Rig 可在两格式中把眼部、刘海或交叉肢体前后画反** | A 用 `DrawOrderPolicy v1` 冻结 back-to-front `part_draw_rank`；B 在 component 出现后按 stable component ID 连续展开 `component_draw_rank`。D/E 只映射 slot/order 方向，default composite 与遮挡 fixture 必须一致 |
| W11 多 mesh component 若作为同一 Spine slot 的多个 setup attachment，只会显示一个 | **成立，是 Spine slot/attachment 基数错误** | v1 固定一 component 一个 Spine slot + setup attachment；同 part slots 连续并共享一个 atlas region，slot 的 setup bone 只作 attachment owner，weighted vertices 仍可引用多骨。Live2D 继续一 component 一 ArtMesh |
| W12 `root` 被定义成 pelvis→spine，半身/头像可能没有任何稳定根骨 | **成立。所有 parent fallback、Spine slot owner 和 Live2D root frame 都会失去共同祖先** | 新增永远存在的 synthetic identity `bone/root`，无 joint、零长度、canvas identity；原 pelvis→spine 改为可选 `lower_torso`。只有 synthetic root 允许空 head/tail，所有缺失祖先最终上提到它 |
| W13 Spine 只写“Y 向上”，没有 canvas→Spine 原点/单位公式 | **成立。整体平移、root 位置和 bind-local vertices 都不确定** | `SpineCoordinatePlan v1` 固定 1 unit=1 canvas px、canvas center 为原点：`(x-W/2, H/2-y)`；synthetic root 输出 `(0,0)`。所有 bone/vertex/landmark 共用 forward/inverse，report 与 runtime fixture 验证 round-trip |
| W14 Live2D root 公式保留 canvas 的 `+y down`，且 MotionClip rotation 正方向未定义 | **成立。Cubism Core 使用 OpenGL 表示，root Y 必须翻转；两 exporter 还可能把同一 rotation 解释成相反方向** | root 改为 `((x-W/2)/PPU,(H/2-y)/PPU)`；Rig transform 固定 translation 为 canvas delta、rotation 为 degree/视觉顺时针正。Spine 映射 `(dx,-dy,-θ)`，Live2D 的 angle/base-angle sign 由 E0 钉住并回到同一 canvas evaluator |

### Revision 15 控制、排序与格式边界收口

| 审查项 | 核对结论 | Revision 15 处理 |
|---|---|---|
| X1 Spine timeline 被写成“显式 linear curve” | **成立。Spine 4.2 JSON 没有 `curve:"linear"` 这个合法编码；线性语义由省略 `curve` 表示** | D 对所有连续 timeline 的线性 segment 固定省略 `curve`；`"linear"`、`"stepped"` 或 Bezier 数组均由 validator 拒绝。MotionClip 中仍显式保存 `interpolation="linear"`，不能把目标格式的省略字段误当成未冻结语义 |
| X2 `component_draw_rank` 无界，却直接映射 Cubism draw order | **成立。Cubism 4.2 的兼容 draw-order 域为 `0..1000`；超过 1001 个 ArtMesh 时无法保持全局唯一顺序** | C 增加逐格式 `FormatModelPlan`；Live2D v1 要求 drawable count `≤1001`，并把从 0 开始的连续 component rank 原值写入 draw order。超限在 C 以 `draw_order_capacity_exceeded` 失败，不能截断、取模或留到 E 猜 |
| X3 semantic priority 只在同 depth bucket 内生效，却宣称固定刘海/五官遮挡关系 | **成立，两条规则互相矛盾；异常 depth 会覆盖本应强制的语义前后关系** | `DrawOrderPolicy v1` 改为显式 behind→front DAG；Kahn 拓扑排序只在当前 zero-indegree 集合内用 depth bucket 与 stable ID 选点。语义边永远优先于 depth，registry 必须全局无环，`<` 明确定义为“先画/在后方” |
| X4 MotionClip channel 数值同时被当成骨骼属性和 Live2D parameter 值 | **成立，是不可实现的数据模型**。一个 idle 控制可同时驱动 torso rotation、head translation 和多个 deformer；它们不可能共用一条既是 degree 又是 pixel 的 parameter curve | 新增 `ControlSpec/ControlCurve/ControlBinding/TargetTransfer v1`。clip 只对 control 写一条曲线；Revision 16 进一步把 target transfer 提升为 Rig 顶层 binding。Spine 求值 transfer，Live2D 每 control/parameter 只写一条 motion curve并把多个 primitive keyform 绑到它 |
| X5 `rig_overrides.json.input_fingerprint` 未说明是否包含 override 本身 | **成立。若包含会形成循环摘要；若含算法/config 又会让同一 canvas 上的人工坐标无谓失效** | 改名并冻结为 `target_input_fingerprint`：只覆盖 see-through manifests、合法 payload、canvas 与 canonical-tag schema，不含 override bytes、auto-rig 算法或运行配置。override SHA 作为独立 stage 输入，二者不得混用 |
| X6 `padding=2` 与 `extrude=2` 没定义 packed rect | **成立。A/C 可以各自得到 `w+4` 或 `w+8`，即使都自称遵守同一 TexturePagePlan** | v1 明确定义 content rect、2px extrusion ring、再外加 2px transparent safety gap；packed footprint 为 `(w+8)×(h+8)`，UV/atlas region 只指 content rect。A dry-run、C pixels 与占用率统一使用 footprint |
| X7 `xmin/xmax` 眼睛被直接映射为 `ParamEyeL/ROpen` | **成立，重新引入了本设计已拒绝的解剖左右猜测**。背面、侧面或镜像角色不能从较小 x 推出 L/R | v1 的分侧眼/眉 control 使用稳定 custom `XMin/XMax` parameter；`EyeBlink` group 可以引用 custom ID。只有未来版本有显式 anatomical-side observation/override 与独立 fixture 时才允许标准 L/R 映射，不能由 E 临时猜 |
| X8 “固定 preset” 只有 ID/capability，没有冻结曲线、时长与几何 transfer | **成立。两个实现可以都通过 schema，却生成完全不同的 idle/breath/head 动作** | 增加 `PresetLibrary motion-core-v1` canonical descriptors：逐 preset 固定 kind、30Hz frames、loop、control keys 和 geometry-normalized transfer 公式；C 物化数值与 descriptor digest。任何调幅/改时长都是 preset version 变更 |
| X9 blink/talk 同时被称为 expression，又被写成 motion3 | **成立，是 artifact kind 矛盾** | v1 固定 blink/talk 为 `MotionClip`（一个 one-shot、一个 loop），happy/sad/surprised 才是静态 `ExpressionPreset`；profile/manifest/artifact key 不能再按自然语言类别猜 |
| X10 motion fade/crossfade 未冻结 | **成立。相同 keyform 在 Live2D 默认 fade 与 Spine AnimationState mix 下会得到不同的首尾轨迹** | `MotionRuntimeApplication v1` 固定资产 parity/默认建议为 weight=1、fade-in/out=0、无 crossfade；motion3 明写零 fade，Spine manifest/harness 使用零 mix。下游自定义混合属于播放器策略，不再宣称与 canonical 单 clip 轨迹相同 |
| X11 Spine bone/weighted bind 只写“转 parent-local”，没有公式 | **成立。parent 旋转存在时，直接减坐标或减角会让骨头与 weighted mesh 在 setup pose 分离** | 增加 `SpineBindPlan v1`：全局 head/tail 先过 CoordinatePlan，再用 parent/bone world-rest affine inverse 求 local head、angle 与每 influence bind vertex；序列化后正向重建所有 joint/vertex做 0.1px parity |
| X12 声称逐字节确定，却只冻结 PNG/Rig JSON encoder | **成立。Spine/model3/motion3/exp3/report 的 key order、float 与换行仍可随实现变化** | 所有公共 JSON 固定 RFC 8785 JCS bytes，schema array order 另有契约；atlas 固定 ASCII/LF grammar，MOC3 固定 little-endian float32/section codec。encoding profile/version 进入对应 stage fingerprint 与 golden tests |

### Revision 16 最终一致性审计

| 审查项 | 核对结论 | Revision 16 处理 |
|---|---|---|
| Y1 `TargetTransfer` 挂在 MotionClip 内，ExpressionPreset 却要求复用 | **成立，是事实源归属错误**。参数到模型目标的映射属于 control/model，不属于某一条时间曲线；只有 expression 使用的 `mouth_form` 甚至没有可供复用的 clip channel | 新增 Rig 顶层 `ControlBinding[]`，每条唯一拥有 `control_id → canonical target/property/transfer`；MotionClip 只拥有 ControlCurve，ExpressionPreset 只拥有 absolute control value。D/E 都从同一 binding 集合解析，禁止在 preset 内复制 transfer |
| Y2 逐 preset 都可实现，不代表它们能同时进入一个 MOC3 | **成立。`talk` 的 `mouth_open` 与 smile/frown 的 `mouth_form` 可分别通过，却可能同时修改同一不可拆 ArtMesh；逐项 preflight 会承诺一个无法合并的模型** | `FormatPresetSetPlan v1` 在 C 做 required-first、版本化 optional-priority 的集合级 primitive union/conflict 检查。required 互冲即失败；optional 冲突按固定顺序 omitted 并记录冲突对象。D/E 只能复算同一 set plan，不能各自挑喜欢的 preset |
| Y3 serializer 已确定，但 component/Delaunay/采样仍可随输入遍历、退化点与 Qhull 改变 | **成立，不确定拓扑会向两种格式传播，JCS 无法修复上游语义漂移** | 增加 A `MaskComponentPlan v1` 与 B `MeshBuildPlan v1`：确定性 cleanup/labels、采样/量化、退化点 symbolic perturbation、canonical vertex/triangle order 和依赖/options fingerprint；不允许随机 `QJ` joggle 或库返回顺序成为 ID |
| Y4 success manifest 与 stage failure record 的关系未冻结 | **成立。失败若覆盖旧成功 manifest 会像可复用 commit，若只留异常对象又无法让 G 稳定汇总** | A-F 的 commit manifest 只表示 `stage_validated*`；失败先撤销旧 commit，再原子写非复用的 stage-local `failure.json`。A-E failure 由 G 复制规范化摘要到公共 error，F failure 只留实验结果。`stage_failed` 是 failure-record 状态，不是 cache hit |
| Y5 startup 声称验证 preset/primitive registry，却没有对应 job-level code | **成立。把打包配置错误塞进首个 item 的 `invalid_motion_clip` 会受 `continue_on_error` 误导** | 增加 `invalid_preset_registry`、`invalid_primitive_registry`、`invalid_profile_registry`；Revision 16 最终审计再补 `invalid_diagnostic_registry`。全部在枚举 item 前失败，不生成 item report |
| Y6 “17 stops” 没有说明作用域，baked-position 计数也未给公式 | **成立。实现可解释为全模型 17、每 preset 17 或每 mesh 17，容量结论不可比较** | 固定为每个唯一 non-rigid `(parameter_id, primitive_target_id)` binding 最多 17 个 stops；总 positions 按 ArtMesh 顶点/warp control-point 的实际 keyform数组求和，去重只允许共享同一 immutable payload digest |
| Y7 draw DAG 只写 base tag，split part 如何展开未定义 | **成立。`eyewhite-r/-l` 若不归到 `eyewhite` 语义边，左右拆分反而绕过眼部顺序** | policy 对 final Part 的 canonical `base_tag` 匹配，规则边展开为实际 Part ID 的笛卡尔积；缺席 tag 不造 phantom node，split suffix 不改变语义层级 |
| Y8 strict avatar 的“调用方允许 procedural/native”让同一 profile 可变 | **成立。版本化 profile 不能把最低能力门留给运行时自由配置** | `dual_runtime_avatar_v1` 固定接受通过质量门的 native/procedural，并明确要求五个 facial preset；若要 native-only 或要求 talk+smile 并发，新增 profile/schema，不能复用 v1 名称 |
| Y9 blink 等 control 同时可能有 native 与 procedural 实现，但 binding 没有互斥分组 | **成立。把所有 applicable bindings 一起发射会同时显示替换层又压 base mesh；让 exporter 自选则重新出现双事实源** | `ControlBinding` 增加 semantic group 与 atomic implementation bundle；C 以固定 `native` 优先、`procedural` 次之选择一个 bundle并写入 format plan，D/E 只复算。bundle 不完整就整体不可用，不能混搭半套实现 |
| Y10 静态 final PSD 不可能同时含“原图未显示的 native 闭眼/笑脸层” | **成立。把额外层塞进 final PSD 会破坏 payload-tag 精确相等，复合时还会污染原图；让 override 指任意外部文件则绕过输入摘要与路径安全** | 新增可选 `NativeVariantSource v1` manifest + 受限 RGBA PNG 目录，独立 `native_variant_set_sha256`；只有 A eligibility/admission 通过的 variant 生成隐藏 render part，且不参与关节。目录缺失表示无 native capability，非法目录硬失败 |
| Y11 一个 native variant 只引用一个 base，无法表达闭眼替换眼白/虹膜/睫毛；把多个 base 强挪成连续组又会破坏遮挡 | **成立。遮挡 support 与插入位置是两种不同关系，不能共用一个 `base_part_id`** | manifest 改为非空 `base_part_ids[]` + 唯一 `draw_anchor_part_id`。base 列表只定义 occlusion coverage；A 保留各 base 的相对 draw order，只把最前 anchor 与 variants 连续展开后重编号。新增冻结 role/base-family/side/component registry 与相应 mutation fixtures |
| Y12 A 因输入非法失败时，canonical input/variant digest 可能尚不可构造，`error.json` 却要求两者必填 | **成立。缺 manifest、非法 JSON 或 symlink escape 会让“记录输入错误”本身再次报错** | 增加不跟随 symlink、只扫描授权候选路径的 `ObservedInputInventory v1`；failure record/error 永远带其摘要，canonical target/variant digest 在未形成时显式为 `null`。成功路径仍禁止 nullable digest |
| Y13 native replacement 同时调低 base opacity，会让 `mouth_open` 与 `mouth_form` 在同一 MOC3 必然争用 base ArtMesh | **成立。即使每个表情各自可播，strict avatar 的 required-set union 仍无法编译；这会把新增素材路径做成假能力** | v1 native 固定为通过覆盖门的 `occluding_overlay_v1`：普通 base 永远 setup opacity 1，只有各自 variant opacity 被驱动。base IDs 只定义应被完整遮住的 support。不同参数可驱动不同 variant target，禁止的 talk+smile 组合由 runtime compatibility matrix 拦截，不再制造模型级共享 target |
| Y14 A 先按全部 variant 做 atlas dry-run，C 才判 bundle/coverage 不 eligible | **成立。一个本可回退 procedural 的坏 optional 素材会先占 atlas，甚至让 core item 以 texture budget 失败；C 若删 region 又破坏 A/C plan identity** | 增加 A-owned、profile-independent `NativeVariantEligibilityPlan v1`，在 packing 前冻结 role bundle、coverage/role-scale/occlusion-intrusion 与 component partition。只有完整 quality-eligible bundle 进入资源 admission，最终 admitted variants 才成为 render Part；B 消费 A 的 partition，C 复核 digest 而不增删 region |
| Y15 合格但非必需的 native bundle 仍可能把 mandatory atlas 或 Cubism drawable budget 挤爆 | **成立。optional capability 不能让只要求 core 的 item 失败；反过来静默降采样、复用 draw order 或拆散 atomic bundle也不合法** | A 先硬验所有 see-through base regions，并冻结 base component count；再按 `blink.native < mouth_open.native < mouth_form.native` 对完整 native groups 逐组重跑同一 packer与 `≤1001` drawable guard。超任一预算只拒绝该 group并记录 warning；最终 admitted set 冻结后再进入 draw/B/C。base 自身的格式容量仍由正式 hard gate 失败 |
| Y16 A 为 admission 数 component，B 又为 mesh 重新 cleanup/split | **成立。两套 threshold/connectivity 只要有一像素差异，A 的 drawable guard、B 的 mesh 数和 C 的 draw order 就不再描述同一模型** | 把 `MaskComponentPlan v1` 明确归 A：所有普通/native masks 只 cleanup、label、side-classify 一次并生成稳定 component IDs。B 的 `MeshBuildPlan` 从 A labels 开始 contour/triangulate，禁止重 threshold；plan digest 进入 A/B/C，组件算法变化从 A 失效 |
| Y17 插入 variant 后又要求所有 base 保持原整数 rank | **成立，是不可同时满足的排序契约**。若 rank 全局唯一且无空洞，在 anchor 后插入节点必然使后续 base 的数值平移；保留原值只能碰撞、留空洞或引入分数 | A 先冻结普通 Part 的相对序列和临时 index，再把 anchor token 展开成 bundle，最后对完整序列从 0 重编号。只承诺普通 base 相对顺序不变，不承诺插入前后的整数相等 |
| Y18 profile/tier 被无差别放进每阶段 config fingerprint | **成立，会重新制造无意义的 A/B 重算，并和 profile-independent admission 冲突** | `relevant_config_fingerprint` 按真实消费面投影：profile 变化从 C 失效，validation tier 变化从 D/E 失效；只有显式升级 A resource-profile/schema 才重跑 A/B。G failure 范围也统一收窄为 A-E，F 保持独立实验 |
| Y19 Spine 静态 expression 只写“hold”却没有冻结时间编码 | **成立。零时长单 key、任意 duration 和依赖 Runtime 对 completed entry 的偶然行为都能声称合规** | 增加 `SpineExpressionHold v1`：每 target 固定 `0` 与 `1/30 s` 两个相同 key，track 1/loop/replace/zero-mix 持续到 clear；Live2D 继续使用无时间轴的 active exp3，并验证 clear 后恢复 base motion |
| Y20 variant eligibility 依赖 component partition，但 MaskComponentPlan 又只处理“已过 content quality”项 | **成立，是 A 内部循环依赖** | 先对所有 manifest-valid candidate 单次构造 A-owned MaskComponentPlan，再做 coverage/side/bundle/resource admission；rejected candidate 只留 A 证据，不进入 final Part/rank/region，B 禁止重分割 |
| Y21 Live2D 产物表仍使用未定义的 `<model>` basename | **成立。两个实现可分别取 item 名、显示名或固定名，都会通过格式 validator 却得到不同路径和 manifest** | v1 artifact basename 固定为 ASCII `model`，三份文件精确为 `model.moc3/model.model3.json/model.cdi3.json`；不得从 item 路径或显示名派生，未来可配置名称必须升级 artifact-layout schema |
| Y22 `page_<index>.png` 没有冻结 index 编码 | **成立。0/1 起始、前导零和稀疏页号都会改变 atlas/model3 路径，却不改变像素计划** | v1 page index 固定为从 0 开始、连续、无前导零的 ASCII decimal；路径精确为 `page_0.png...page_{N-1}.png`，三个目录复用同一 basename |
| Y23 摘要算法已冻结，但摘要字符串编码没有冻结 | **成立。大写 hex、裸 hex、`sha256:` 前缀和 base64 都能表示同一值，却产生不同 JCS/cache bytes** | 新增 `DigestEncoding v1`：JSON/API 一律 `sha256:` + 64 lowercase hex；以 digest 命名的文件只用 64 lowercase hex、无前缀；拒绝大写、base64 和截断值 |
| Y24 GlobalExportSymbolTable 要求“稳定 hash suffix”，却没有名称 codec | **成立。清洗字符、截断长度、冲突时谁保留裸名、hash 输入/长度任一不同，都会产生两份各自稳定但互不兼容的包** | 新增 `InternalIdCodec/ExportNameCodec/SymbolKindCodec v1`：内部 ID 使用 typed prefix + lowercase ASCII slug/完整 identity digest；generic stem、逐 family base/token、63-byte 截断与 16-hex typed-key suffix 全部冻结，同一碰撞类成员同时加 suffix，残余碰撞硬失败 |

### Revision 17 实施与输入质量复审

| 审查项 | 核对结论 | Revision 17 处理 |
|---|---|---|
| Y1 `finalize_success` 只接受 A-E expected fingerprints，`is_item_completed` 却把同一 mapping 当成 A-E/G 完整集合 | **成立，已复现；这会让所有合法 completed item 永远无法命中 `skip_completed`** | terminal API 统一只接受且严格校验 A-E。`StageGraphValidator` 对 G 完整重哈希和递归校验，但 G 的 freshness 不再要求调用方提供一个无法独立推导的 expected fingerprint；`is_item_completed` 由 export payload、当前 C/D/E manifests/artifact sets 与 terminal config fingerprint 复算 G 语义 |
| Y2 foundation plan 钉在 Revision 13，manifest 没有 W3 的精确 inventory | **成立；仅验证已声明文件会保留旧 page/motion/expression** | plan 升到 Revision 17；manifest schema 升版并加入 `output_inventory_sha256`。C/D/E/G public owner namespace 必须与声明集合精确相等，提交时清除 obsolete 文件，resume 对任一未声明 public artifact 返回不可复用 |
| Y3 原子替换顺序被误当作崩溃安全本身 | **成立；未对父目录 fsync 时，rename 顺序不能证明断电后的目录持久性** | 冻结不变量：恢复安全来自每次 resume 重新读取并哈希 manifest 与全部 artifact；mtime/size cache 不能替代 byte hash。commit-marker-last 只缩小撕裂窗口，撕裂状态仍必须因摘要不符而重跑 |
| Z1 `spill_mass` 的半径随 base bbox 放大，测不到近邻五官遮挡 | **成立，但“所有其他 amodal mask 的 raw union”也会把眼睛下方合法的 `face` support 误判为侵入** | Revision 17 先引入 draw-prefix visible contribution；其最初的 `min(mass_V,mass_B)` 分母已在 Revision 18 被 role-scale + 真实 intrusion fraction 取代。旧 distance spill 只保留诊断，不再决定 eligibility |
| Z2 DrawOrderPolicy 对 earwear 有硬边，却漏掉 eyewear | **成立；眼镜与眼部近共面，Marigold depth 不应覆盖语义关系** | 增加 `{face,eyewhite,irides,eyelash,eyebrow} < eyewear`。`headwear` 与 front/back hair 没有跨角色通用拓扑，v1 明确保留 depth/stable-ID 决策，不伪造一条必错的全局边；未来只有显式关系输入/override 才能升级该策略 |

### Revision 18 NativeVariant role-scale 复审（授权域已由 Revision 19 取代）

| 审查项 | 核对结论 | Revision 18 处理 |
|---|---|---|
| AA1 `min(Σa_V,Σa_B)` 假设 variant/base alpha 规模相近，使 mouth native 数学上不可达 | **成立。闭嘴 base 是线，张嘴 replacement 是面；这个分母把合法扩张本身计成数百倍侵入** | 拆成独立 `alpha_mass_ratio≤k_role` 与 `occlusion_intrusion=Σ(a_Va_U)/Σa_V≤0.01`。role registry 冻结 `eye_closed=1.5`、`mouth_form=4.0`、`mouth_open=12.0`；scale gate 防稀释，intrusion 保持为真实分数 |
| AA1 扩张型 mouth 的 face authorization 仍锚在 base support | **成立。mouth variant 合法地必须覆盖闭嘴线之外的 face 像素** | Revision 18 曾改为授权 variant 实际 alpha support 下的 face contribution；AB1 证明这是自指恒真条件，现行契约已在 Revision 19 撤销该分支 |
| AA2 `ceil(canvas_edge/128)` 用画布尺度代替被测特征尺度 | **成立。letterbox 使相同 canvas 上的特征线性尺度可相差数倍** | Revision 18 先把 blink 改为 feature-relative radius；Revision 19 又曾让 mouth 改读 cleaned base support span，这一中间尺度已由 Revision 20 的共同 `sqrt(mass_B)` estimator 取代 |
| AA1/AA2 的具体常数缺真实批次证据 | **成立；结构可冻结，数值仍必须接受生产样本证伪** | 常数作为 v1 product policy 进入 registry digest，不允许运行时自调；正式启用 native adapter 前按 role、768/1024/1280 与主体尺度分桶报告 ratio/radius/rejection 分布。门不通过只能升 registry/profile version，不能现场放宽 |

### Revision 19 NativeVariant 授权域复审（尺度估计器已由 Revision 20 取代）

| 审查项 | 核对结论 | Revision 19 处理 |
|---|---|---|
| AB1 expanding role 在 `support(a_V)` 下授权 face，使 face intrusion 在 `a_V` 加权域内恒为零 | **成立，是自指授权，不是阈值问题** | 删除 variant-derived authorization。所有 role 都只允许 `face ∩ dilate(support(a_B),R_role)`；授权域完全由 base/registry 决定，`a_V` 只作为被测量对象出现 |
| AB1 建议 expanding 也回到 base dilation | **方向正确，但继续使用 `sqrt(mass_B)` 仍会让细闭嘴线得到过小半径** | preserving eye 使用 alpha-mass 等效尺度；expanding mouth 使用 cleaned base support 的长轴 span。mouth-form 固定 `c=0.25,r=4..24 px`，mouth-open 固定 `c=0.60,r=8..48 px`，再统一走同一个 Euclidean dilation/visible-intrusion kernel |
| AB2 `k_role` 只限制总量，不限制十二倍 alpha 往哪边生长 | **成立；scale gate 单独不能提供空间锚** | base-anchored envelope 成为空间上界：授权半径外的 face 与全部未授权 part 都进入 intrusion。新增 `mass_V=11×mass_B` 但主体长到 envelope 外的 cheek fixture，必须失败；位于 role envelope 内的合法 mouth-open 才能通过 |
| expanding role context 曾被规定为 `null` | **随自指分支删除而失效** | expanding row 现在必须提供 scale estimator、`c/r_min/r_max`；缺字段、仍写 `null` 或由 variant geometry 推导半径均为 `invalid_primitive_registry` |

### Revision 20 NativeVariant 尺度与诊断复审

| 审查项 | 核对结论 | Revision 20 处理 |
|---|---|---|
| AC1 `sqrt(mass_B)` 与 bbox 长轴虽然同为 pixel length，却不是同一种统计尺度，`c_role` 不能跨 role 直接比较 | **成立；严格说是估计器语义不一致，而不是物理量纲不同** | 删除 mouth-only `support_span_B` 分支。全部 role 固定 `feature_scale_B=sqrt(mass_B)`；unclamped 区间内 `c_role` 重新具有同一尺度上的倍率语义。v1 固定 eye/form/open 的 `c=0.20/1.00/2.00`，clamp 保持 `2..12/4..24/8..48 px` |
| bbox 长轴会被 cleaned support 中残留的远端像素或细长组件放大 | **成立；换成共同 alpha-mass 尺度可消除“远端像素拉长全局 radius”的路径** | 增加同 alpha mass、不同 bbox span 的 radius-invariance fixture。注意 `r_min` 只处理绝对小特征，不能证明任意高长宽比 mouth 都够宽；正式 gate 必须按 cleaned support aspect ratio 分桶验证，失败只能升级 envelope version |
| AC2 `face_alpha_outside_envelope` 只解释 face 分量，无法解释 nose/eyebrow 等导致的同一 intrusion failure | **成立，单一 face 字段会让合法的总 gate 与诊断表象矛盾** | 删除该字段，改为完整、稳定排序的 `intrusion_by_part[]`，并同时记录 `intrusion_face_ratio`、`intrusion_other_parts_ratio`；三者必须与 `occlusion_intrusion` 闭合，不截断 top-N |

### Revision 21 官方 Core 与 V4.00 golden 实测

| 实测项 | 核对结论 | Revision 21 处理 |
|---|---|---|
| XKLive 随附 `Live2DCubismCore.dll` 是否可用 | **可用于开发期 Core runtime probe**。该 x64 DLL 的 Authenticode 签名有效，SHA-256 为 `e20e8364850e4c0b726566237855c3b359ad946b10e7f628f4ddad15bbd3730e`，`csmGetVersion()` 返回 `04.02.0002`，`csmGetLatestMocVersion()` 返回 `4`；官方 `CubismWebSamples` `4-r.4` 的 Hiyori V4.00 MOC 可完成 aligned revive、model initialize、首次 update 和有限顶点读取 | 新增 opt-in `probe_cubism_core()` 与崩溃隔离的 `exercise_moc_with_core()`；probe 只记录 binary identity/API/runtime evidence，不自动生成 attestation，也不把本机 DLL 打包进 wheel |
| 该 DLL 能否承担正式 consistency gate | **不能**。导出表没有 `csmHasMocConsistency`；官方 change history 说明该 API 在 `04.02.0003` 才加入，而 `04.02.0004` 又修复了错误拒绝及 malformed MOC 触发 crash 的问题 | consistency gate 的最低已知安全 floor 固定为 `04.02.0004`。达到 floor 只表示“可进入 E0 attestation 候选”，仍须按 binary SHA-256 实测 allowlist、SDK/Viewer render gate 与许可证审计，禁止把 semver floor 等同于 release approval |
| 结构 parser 把所有 SOT section offset 强制 64-byte 对齐 | **错误**。`csmAlignofMoc=64` 约束传给 Core 的 MOC 内存基址，不是文件内每个 section；官方 Mark V4.00 golden 的合法 SOT offset 包含 `2312`，现有 blanket rule 会误拒官方文件 | envelope validator 只对 required offset 做 non-zero/in-bounds/nondecreasing，对 unused offset 做 zero-or-in-bounds；每个 body section 的实际 packing/alignment 由后续 typed-section codec descriptor 单独冻结。`moc_buffer_alignment=64` 与 `final_file_alignment=64` 保留为不同字段，禁止再命名为 `body_alignment` |
| 当前实测是否足以签署 `live2d-frames-v1.json` | **不足**。Core 本身不提供 drawing，且本轮只加载官方既有 MOC，没有验证本项目 writer 的 nested warp/rotation、UV/alpha、motion/expression 或 default-rest parity | 不生成 production attestation。Hiyori/Mark/Rice golden 仅作为 parser/runtime provenance；下一 E0 slice 仍必须实现最小 writer，并用 `04.02.0004+` allowlisted Core 与官方 SDK/Viewer 完成固定矩阵 |

### Revision 22 官方 SDK 5-r.5 与项目 writer E0 实测

| 实测项 | 核对结论 | Revision 22 处理 |
|---|---|---|
| 官方 Core 候选是否达到 release 技术门 | **达到首个精确 binary allowlist**。`Core/dll/windows/x86_64/Live2DCubismCore.dll` 返回 `06.00.0001`、latest MOC `6`，包含 `csmHasMocConsistency`，SHA-256 为 `d883c00d114fdf6cef61f439feb23e02d000fdf683e092803010470b80dfaf09`，Authenticode 状态为 Valid | allowlist 只加入该 `windows/x86_64/SHA-256` 三元组，不把 `6.x` semver 范围整体放行；SDK/Core 不复制进仓库、wheel 或导出资产 |
| 项目 V4.00 writer 是否能通过官方 runtime | **通过**。typed codec 可 decode/re-encode 官方 V4 goldens；项目 static/deformer/negative-default 三个 MOC 均通过 consistency、revive、initialize/update。zero-length section 的 SOT 允许恰好等于 EOF，非空 extent 仍由 typed codec 拒绝越界 | envelope descriptor 升为 `moc3-v400-envelope-layout-v3`，另以 `moc3-v400-sections-v1` 钉住 100 个 typed section、count source、element size 与 writer alignment；MOC header 继续输出 V4.00 version byte `3` |
| 嵌套 warp/rotation 的实际坐标语义 | **在 v1 限定域内通过**。`root→rectangular warp(quad_transforms=true)→rotation→rotation→ArtMesh` 的 rest、端点、九点 angle+origin 插值与多参数组合均在 canvas `≤0.1 px`；反转两个不同 pivot 的 rotation stack 产生可见差异。warp 会移动 rotation pivot，但不把自身非均匀伸缩 Jacobian 继续乘到 rigid local vector | `live2d-frames-v1` 只签署 axis-aligned rectangular structural warp；不得把 `quad_transforms=false` 当作普通 bilinear，也不得把本结果外推到任意曲面 warp。一般 warp/local inverse 需要另立 E0 与 schema version |
| default-rest 是否只靠结构 consistency | **不能**。正例把 `ParamInner.default=6` 显式写为非中点 rest key，初次 update 还原 rest；负例保留同一 default 却遗漏对应 rest key，结构仍可加载但顶点偏差显著 `>0.1 px` | attestation 同时钉正/负 fixture；每个正式模型仍需逐 item 做 default 参数后的 vertices/opacity/draw-order parity，不能以 `csmHasMocConsistency=true` 替代 |
| MOC、Core API 与最终 D3D11 纹理 V 方向 | **是三层不同表示**。MOC `uv.xys` 使用 canonical top-left V；Core drawable API 返回 `1-v`；官方 D3D11 shader 再翻一次 V，最终采样回到 canonical top-left。四角异色 + 半透明中心 fixture 已证明最终方向和 straight-alpha loader | 新增隔离 `uv_kernel.py` 并纳入 frame digest；共享 page PNG 不再被误读为 Spine/Cubism 数值 UV 相同。E0 同时保存 parser/Core/render 三层证据 |
| motion3/exp3 能否按既定顺序生效 | **通过**。base parameters 后先把 motion 采到 1 秒，再应用 zero-fade exp3；Add/Multiply/Overwrite 分别得到 `0.75/15/-8`，且离屏像素摘要变化 | D3D11 WARP harness 报告 observed parameter values 与 raw RGBA；exp3 不在默认 fade 第一帧盲断言，正式 dual-runtime 表情能力仍只采用 Overwrite，Add/Multiply 只作 Live2D writer conformance |
| 官方 Framework 是否接受 JCS/minified motion JSON | **不可靠**。5-r.5 Framework numeric parser 只把逗号或换行识别为数字终止符，合法的 `10.0]`/`3}` 紧凑 JSON 会被拒绝 | JCS 只用于摘要/attestation；`.motion3.json`、`.exp3.json` 固定使用 ASCII、`indent=2`、stable key order 和末尾换行的 `cubism_runtime_json_bytes()`。禁止为了体积复用 JCS bytes 作为 runtime 文件 |
| 技术 attestation 是否等于商业发布许可 | **不是**。Framework 受 Open Software License，Core 仍受专有条款和收入/用途条件约束 | generator 只在显式确认“不随包分发 SDK/Core，组织许可判断是外部门”后写技术 attestation；正式发行前仍需由发布主体完成许可证判断，本规范不提供法律结论 |

### Revision 23 Stage A 左右状态实现复审

| 审查项 | 核对结论 | Revision 23 处理 |
|---|---|---|
| A 复用上游 `V3_SPLIT_FAMILIES` 决定哪些 mask 可按可靠双组件推导左右 | **错误。** `legwear/footwear` 在 v3 不能带 `-r/-l` 后缀，但这不代表其两个可靠连通域没有 image-side 证据；直接复用会让腿、脚永远不可能成为 `merged-separable` | 拆成两个 registry：canonical tag registry 继续只约束合法上游 suffix；`MaskSideClassifier v2` 的 geometry-side family 为 `V3_SPLIT_FAMILIES ∪ {legwear, footwear}`。只有恰好两个通过面积门且质心 x 可严格排序的组件才生成 `.xmin/.xmax`，否则保持 `merged-ambiguous` |

### Revision 24 Stage A 关节证据实现复审

| 实现项 | 核对结论 | Revision 24 处理 |
|---|---|---|
| geometry factor 的 `branch_ratio` 是否可反向写成“无支路置信度” | **不可以。** 字段名表达的是次长/最长 geodesic endpoint path 的原始比值；反写 `1-ratio` 会让阈值、报告与字段语义互相矛盾 | `limb-joint-geometry-v1` 保存原始 `[0,1]` 比值，值越高表示端点竞争越强；是否通过 hand-tip/toe gate 由 descriptor 阈值单独决定，不把它伪装成总置信度 |
| merged limb 的 eligibility 是否可继续引用不存在的 sided mask | **不可以。** 这样 pose/override anatomy gate 永远找不到可验证 support | `merged_limb` 证据改为真实的 `mask/limb/<family>.merged`；它仍不允许 geometry 强拆双侧，但可供显式 pose/override 做 support 校验 |
| override 与 `missing` eligibility 的优先级 | resolver 层继续兑现“合法 override 无条件优先”；但 Stage A 在进入 resolver 前先校验 target identity、anatomy support 与 `allow_outside`。没有显式 opt-in 的无 support override 直接拒绝 | `observation-anatomy-validator-v1` 固定 evidence-mask 距离门；显式 `allow_outside=true` 的 joint 无论最后是否仍落在 support 内都写入 warning set，避免人工授权在缓存/报告中消失 |
| pose backend 是否成为默认依赖 | **否。** provider 输出必须封装为 `PoseObservationBatch v1`，携带 provider/preprocess fingerprint、anatomy plan digest 与原始 model score；batch 缺失即生成确定性的 disabled identity | `StageAJointPlan v1` 合并 axial、limb、pose、override 原始 observations 与全部 resolved/unresolved/missing 结果；validator 重算嵌套摘要、source projection 和 resolver 结果，exporter 只消费该计划 |

### Revision 25 Stage B 几何实现复审

| 实现项 | 核对结论 | Revision 25 处理 |
|---|---|---|
| 缺失 wrist/ankle 时是否还能机械照抄完整 BoneSpec 树 | **不可以。** 不存在的中间骨既不能伪造零长骨，也不能让真实后代引用断裂 parent | `BoneGraphPlan v1` 只从 `StageAJointPlan.resolutions` 发射可验证骨；缺失节点导致对应骨省略，仍可发射的后代提升到最近已发射祖先。override 产生的零长/超画布骨严格失败，普通几何退化则稳定省略并诊断 |
| B 是否可从 PNG/alpha 重新阈值或依赖 Qhull 随机 joggle | **不可以。** 这会使 A/B 对 component 身份产生两套真相，并让退化点集跨版本漂移 | `MeshBuildPlan v1` 只消费经过摘要认证的 A 阶段 QCL component；边界/内部采样、`1/256 px` 量化、`<1/4096 px` 符号扰动、轮廓/顶点/三角形排序全部冻结，Qhull 禁用 `QJ`。凹区、孔洞和跨 component 三角形按 alpha support 复验，退化 component 产出具名诊断而非随机网格 |
| 权重是否可继续使用固定像素宽度和全骨候选 | **不可以。** 固定 `proj/40` 随分辨率改变语义，全骨候选会把脸/头发错误绑定到手臂 | `SkinningPlan v1` 先用 Part semantic registry 限定候选骨，再把 limb 顶点投影到 resolved joint polyline 的累计弧长；transition band 来自 joint eligibility radius，权重定点量化、最多四项、按 bone ID 排序并确定性归一化。候选不足时使用可报告的 rigid fallback，不伪装成完整多骨能力 |
| B cache 是否可以是以后由 C 补字段的半成品 `rig.json` | **不可以。** C 改写后会永久破坏 B output digest，使昂贵 mesh/weights 每次 resume 都重跑 | `ComponentDrawOrderExpander v1` 冻结 `(part_draw_rank, component_id)` 的 gapless rank；`RigGeometryCache v1` 嵌入经摘要认证的 A/B plan 投影并验证 parent/joint/component/mesh/influence 引用闭包。B manifest 只拥有 `rig/cache/B/rig_geometry.json`，mesh descriptor 改变只使 B 及其下游 C 失效，A 保持可复用 |
| 实现回归是否覆盖官方 Cubism 环境和上游 see-through 边界 | **是。** 不能只依靠 Stage B focused fixture | 官方 SDK/Core 环境下 auto-rig：`521 passed, 4 skipped`；Stage B focused：`47 passed`；see-through：`54 passed`；dependency/uv：`193 passed`。Ruff、`compileall` 与 `git diff --check` 同时通过 |

### Revision 26 Stage C 模型与事务实现复审

| 实现项 | 核对结论 | Revision 26 处理 |
|---|---|---|
| C 是否可在 capability 过滤后再枚举 primitive/symbol，或让两个 exporter 各自命名 | **不可以。** 这会让罕见 binding 到生产批次才暴露 `missing_export_symbol`，也会使同一 internal identity 随格式和剪枝结果漂移 | `PrimitiveCandidateEnumerator v1` 先按完整 Rig 与 registry 枚举 profile-independent typed-key 超集；`GlobalExportSymbolTable v1` 从该超集单次生成，按 `(format, namespace)` 消解名称。属性测试覆盖全部 profile/preset 的 `binding_plan_keys ⊆ candidate_universe`，exporter 内二次 sanitize 被拒绝 |
| 公共 `rig.json` 是否可由 B 写半成品、再由 C 补字段 | **不可以。** 共享路径会破坏 B manifest 摘要并废掉昂贵 mesh/weight resume | B 只拥有 `RigGeometryCache v1`；C 从经摘要复验的 A/B cache 一次组装完整、引用闭合且 canonical-JCS 的 `RigDocument v1`。capabilities、bindings、clips、expressions、texture pages、primitive candidates 与 export symbols 均在同一事务中冻结 |
| Spine/Live2D feasibility 是否可以埋进 writer，失败后再回改公共 Rig | **不可以。** writer 才发现 required capability 缺失会制造格式间能力漂移和部分发布 | C 在写任何格式产物前生成纯 `FormatPlan`：required preset 双格式严格对称，optional preset 按 `supported_formats/omitted/reason` 明示；Live2D v1 的无 Glue joint bend 固定 omit，Spine 可保留 wave。D/E 只能执行已冻结决定，不能反推或改写 |
| canonical texture page 是否仍由 D/E 各自编码 | **不可以。** 相同像素经两个 PNG encoder 仍可能因 filter/zlib/chunk 变化而字节不同 | C 复算 A 的 pack plan 并要求逐字段一致，再单次编码 `rig/shared/textures/page_<index>.png`；D/E 后续只能字节复制。C manifest 拥有精确 public inventory，失败先移除旧 commit marker，禁止部分公共成功态 |
| Stage C 是否已经等于双格式正式交付完成 | **否。** C 只冻结 exporter 的全部输入与判决，不生成目标 runtime 包 | D 的 Spine 4.2 encoder/runtime validation、E 的 MOC3/model3/motion3/exp3 compiler/runtime validation，以及 G 终态发布仍是后续实现 gate；Revision 26 不写虚假 `completed` marker |
| 实现回归是否覆盖官方 Cubism 环境和上游边界 | **是。** SDK 路径必须实际进入 Core/E0 测试，而不是只检查文件存在 | 官方 SDK/Core 环境下 auto-rig：`579 passed, 4 skipped`；Stage C focused：`60 passed`；see-through：`54 passed`；dependency/uv：`197 passed, 1 skipped`。auto-rig Ruff、`compileall` 与 `git diff --check` 同时通过 |

### 决策摘要

新增 `module/auto_rig/`，消费 see-through item 目录，生成版本化 `RigDocument`，并从同一份
Rig 批量导出带预设动作/表情的 Spine 4.2 包和 Live2D Cubism runtime 包。它不是编辑器、
不是预览工具，也不是 see-through 内部第四阶段；不生成 GIF/WebM，也不把交互式预览作为
产品功能。

实现分七个具名 stage gate；它们可独立测试和缓存，但不是七个可独立发布的产品：

1. **A：输入适配 + NativeVariant eligibility/component partition + 掩膜几何 + 关节质量报告 + overrides**；
2. **B：网格 + 通用多骨骼权重 + 私有 `RigGeometryCache v1`**；
3. **C：冻结 ControlSpec/ControlBinding/曲线/target transfer，执行能力驱动的 exporter-neutral 动作/表情 preset 绑定 + 逐格式 model/preset/preset-set 纯 feasibility preflight + Rig 级全局符号表 + 共享纹理页规划/canonical PNG，并组装/写入完整 `RigDocument v1`**；
4. **D：Spine 4.2 setup rig、animations、简单多页 atlas 导出与验证**；
5. **E：Live2D MOC3 V4.00 runtime、motions、expressions、纹理导出与验证**；
6. **F：可选姿态后端评测，SDPose-OOD Body 为首选候选、RTMW-l 为备选，不阻塞 A-E/G**；
7. **G：item terminal finalization；正式 release 成功时验证并发布双格式 `export_manifest.json`，任一已枚举 item 的 A-E production-stage 失败时发布公共 `error.json`**。

姿态模型默认关闭。没有几何基线和人工标注评测集之前，不把“SDPose-OOD 替换 DWPose”
写成既成事实。正式批处理 profile 固定要求 Spine 4.2 与 Live2D runtime **同时成功**；单格式
开关只用于开发诊断，不能把只成功一半的 item 标为 completed。正式 release 的依赖图是
`A → B → C → (D, E) → G`；任一 A-E failure 也走 failure edge 到 G 写终态错误。D/E 可并行验收，
但 E 未完成时项目就尚未达到正式交付定义。F 是实验分支，不在 release 关键路径。
`spine_4_2_dev` 的成功图只到 `A→B→C→D`；dual profile 配合 structural tier 时成功图是
`A→B→C→(D,E)`，在 D/E 两份 stage report 都完成后结束。两者都没有 G success edge。它们的任一
A-E item failure 仍有 failure edge 到 G，以保持公共
`error.json` 的单写者和批量可寻址诊断。

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
5. 一条 `.motion3.json`、在三个互不 clamp 的参数上分别覆盖 `Add/Multiply/Overwrite` 的零 fade
   `.exp3.json` fixture（可同文件，必须逐参数观测）和一张纹理页；
6. 纹理页四角和中心使用五种不对称标记色，其中至少两处是已知 straight-RGBA 的半透明颜色，
   ArtMesh UV/三角形也不做轴对称布局；同时核对 Core 返回的 UV、parser 解析值与最终渲染采样，
   签署 `CubismV400UvAdapter` 的 page origin、U/V 方向、region offset、边界取样和 target-specific
   alpha loader mode。只看到“纹理非空”或只测全不透明像素不能通过该项。

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

Revision 21 的 `04.02.0002` probe 仍只保留为低版本负例；它没有 `csmHasMocConsistency`，不得进入
release allowlist。Revision 22 已用官方 SDK 5-r.5 的 Core `06.00.0001` 完成完整 E0，并按精确
SHA-256 加入首个 allowlist。`04.02.0004` 继续只是 consistency API 的最低已知安全 floor，不代表任何
满足 semver 的 DLL 自动通过；每个新增 binary 仍须逐 SHA 重跑同一 E0 validator protocol。Core 只计算
模型数据、不含 drawing，因此非空渲染、UV/straight-alpha 和 motion/expression 像素变化由官方
Framework D3D11 WARP harness 独立证明，不能拿 `csmUpdateModel()` 成功冒充渲染通过。

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

可选的 native 表情素材**不属于**上述 see-through PartSource，也不能靠在 final PSD 里偷偷多放一层绕过
tag 精确相等。v1 只接受一个独立的 `NativeVariantSource`：

| 文件 | 契约 |
|---|---|
| `rig_inputs/variants/manifest.json` | `NativeVariantManifest v1`；逐项记录稳定 `variant_id`、`semantic_role`、`composite_mode="occluding_overlay_v1"`、非空 `base_part_ids[]`、唯一 `draw_anchor_part_id`、canvas `xyxy`、相对 PNG path、RGBA/alpha/color contract 与 file SHA-256 |
| `rig_inputs/variants/<variant_id>.png` | cropped straight-alpha sRGB RGBA，alpha mass 必须为正；尺寸严格等于 manifest `xyxy`，只用于隐藏 setup variant，不冒充 depth/joint evidence |

manifest 的 path 必须精确等于 `<variant_id>.png`，规范化后仍位于 `rig_inputs/variants/`；该目录除
`manifest.json` 和 manifest 精确列出的 PNG 外不得有其他 filesystem entry。拒绝绝对路径、`..`、任何
symlink/junction/reparse point、
重复 ID/path、未声明/缺失文件、SHA/尺寸不符和未知 semantic role。`variant_id` 固定为小写 ASCII slug
`[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?`，并拒绝大小写无关的
`con/prn/aux/nul/com1..com9/lpt1..lpt9` Windows reserved basename；`base_part_ids` 必须先证明无重复，再按
stable Part ID 排序参与摘要，数组输入顺序不具语义；实现不能静默 dedupe。
admitted entry 的 render Part internal ID 精确为 `part/native.<variant_id>`；`part/native.` namespace 在 v1
只属于该 adapter，普通 see-through tag/Part 不得占用，`variant_id` 本身继续作为 source identity 保存而不
冒充 Part ID。derived ID 若与任何普通/admitted Part 重复则 `input_contract_mismatch`，不能按发现顺序加数字。
所有 base 与 anchor 都必须是当前 see-through payload 中存在的普通 Part，anchor 必须在 base 列表内，且在
不含 variant 的 canonical draw order 中恰为这些 base 的最前层；不存在、跨角色族、引用另一个 variant、
anchor 不是最前层或 base 的 bone-eligibility class 不一致都以 `input_contract_mismatch` 失败。
role registry 还从当前普通 Part/component 集合推导 `expected_base_part_ids`：eye role 必须覆盖目标 image side
上现存的全部 `{eyewhite,irides,eyelash}` parts，mouth role 必须覆盖现存全部 `mouth` parts；manifest 列表必须
与 expected set 精确相等，不能少列 iris 来让 coverage 数字虚高，也不能多列 eyebrow/face 扩大遮挡范围。

variant 没有深度；它只从 anchor 继承 `base_tag/depth_bucket/bone eligibility` 用于 draw/rig attachment，
`base_part_ids` 只定义必须被 overlay 遮住的 base support，不能把多个 base 的几何/深度合并进 variant，也
不能生成调低普通 base opacity 的 binding。variant 自己的
alpha mask 独立生成 mesh。每项固定 `setup_visibility="hidden"`、
`draw_position="immediately_above_anchor"`。manifest/PNG 的 canonical
`native_variant_set_sha256` 作为独立 stage input；目录缺失表示没有 native capability，目录存在却无效则
`input_contract_mismatch`，不能静默退回 procedural。

`NativeVariantRoleRegistry v1` 只允许以下 role，且是 primitive registry 的组成部分：

| `semantic_role` | 合法 base family | variant-opacity control 语义 |
|---|---|---|
| `eye_closed.xmin` / `eye_closed.xmax` | `base_tag ∈ {eyewhite, irides, eyelash}` 的对应 image-side components | 分别绑定 `control/eye_open.xmin` / `.xmax`，`variant_opacity=1-control` |
| `eye_closed.coupled` | 同一组 eye base；variant alpha 必须恰有两个可按 x 排序的可靠组件 | C 展开成 xmin/xmax 两个 sibling branch，各自由对应 eye-open control 独立控制；组件数不是 2 即 native implementation 不 eligible |
| `mouth_open` | `base_tag=mouth` | `variant_opacity=control/mouth_open` |
| `mouth_smile` / `mouth_frown` | `base_tag=mouth` | 分别为 `max(control/mouth_form,0)` / `max(-control/mouth_form,0)` |

同一 registry 还冻结 `NativeVariantRoleEnvelope v1`；`alpha_mass_ratio` 使用完整 canvas 上的 alpha mass，
不是 support bbox 面积。所有 role 共享唯一的
`feature_scale_kind="sqrt_base_alpha_mass_v1"` 与 `feature_scale_B=sqrt(mass_B)`，role row 只提供同一尺度上的
无量纲 multiplier 和 pixel clamp：

| role family | support mode | `k_role` | `c_role` | radius clamp |
|---|---|---:|---:|---|
| `eye_closed.*` | `support_preserving` | `1.5` | `0.20` | `r_min=2 px`，`r_max=12 px` |
| `mouth_smile` / `mouth_frown` | `support_expanding` | `4.0` | `1.00` | `r_min=4 px`，`r_max=24 px` |
| `mouth_open` | `support_expanding` | `12.0` | `2.00` | `r_min=8 px`，`r_max=48 px` |

因此只在未触发 clamp 时，三个 `c_role` 才能直接解释为同一 base-equivalent length 上的 `1:5:10` 半径倍率；
clamp 后必须报告最终 pixel radius，不能继续用倍率推断实际 envelope。
registry 序列化不用二进制浮点自由值：三个 `k_role` 分别存成 `{3,2}/{4,1}/{12,1}` 的
`{numerator,denominator}`，三个 `c_role` 分别为 `{1,5}/{1,1}/{2,1}`。`mass` 由原始
`alpha_u8` 的整数和除以 255 定义；scale 比较用整数交叉乘法。square-root/round-half-up 使用精确有理
不等式或固定 Decimal context；禁止依赖平台 `float` 恰好落在半整数哪一侧。base support 或 variant alpha
为空时不是一个可计算的 envelope，candidate 必须拒绝：全透明 variant 违反 PNG alpha contract，报
`input_contract_mismatch`；cleaned base component/support 为空则报
`native_variant_ineligible(reason=component_partition)`。不能依靠公式中的 epsilon 或 `r_min` 把空素材伪装成
合法结果。

`mouth_form.native-v1` 的两个端点分别过同一 `k=4.0` 门，任一个失败仍按 atomic bundle 整体拒绝。
`eye_closed.coupled` 也必须先按 A 冻结的两个 x-ordered component branches 分别计算 `mass_B/mass_V`、
radius、coverage 与 intrusion，再要求两侧都通过；禁止把双眼 mass 合并后用 `sqrt(total)` 放大每一侧 context。
共同 scale kind、support mode、全部常数、rounding 和 authorization policy 都进入 role-registry digest；缺项、
NaN、非正 `k_role/c_role`、`r_min>r_max`、role row 试图覆盖共同 scale kind，或任何 role 从 variant geometry/
bbox span 推导授权半径，都在 startup 以 `invalid_primitive_registry` 失败。eligibility record 对所有 role 都写非负整数
`base_context_radius_px`；`null`、临时默认值或 variant-derived radius 都非法。

单侧 eye role 不能引用另一侧 component；coupled role 也不能把两个组件永久绑成一个运行时参数。role 表、
side/component selector、base-family 约束和 visibility branch template 进入 primitive-registry digest；built-in
表缺行、重复 role/control 映射或 branch template 不闭合时在 job startup 以
`invalid_primitive_registry` 失败，未知输入 role 则是 item 的 `input_contract_mismatch`。
admission priority 表必须与 registry 导出的 native atomic group 集合精确相等且 rank 全局唯一；v1 集合/顺序
就是 `{blink.native, mouth_open.native, mouth_form.native}`，漏项、重复或新增未排序 group 同样是
`invalid_primitive_registry`，不能回退字典序。

同一角色 v1 每个 role 最多一项；blink native coverage 必须恰为 `{eye_closed.xmin, eye_closed.xmax}` 或
`{eye_closed.coupled}` 二选一，不能两套并存。coupled entry 必须在 A 的 `MaskComponentPlan` 中稳定得到两个
x-ordered components；C 只能在 B 已把这两个 component IDs 物化为 mesh 后，展开成两个
component-level visibility targets。单侧 entry 也只选择该侧 components，绝不能为了 opacity 方便隐藏整个未拆
Part。`mouth_form.native-v1` 是包含 smile 与 frown 两个端点的 atomic implementation：两项必须引用相同
base 列表/anchor，少一个就整套不 eligible；`mouth_open` 是独立 implementation。duplicate role、重叠 eye
coverage 或不一致 mouth-form base/anchor 属于无歧义的 `input_contract_mismatch`，不能由 C 挑一个“最像的”。

`NativeVariantCompositePlan v1` 不接受 transparent replacement sprite。把列出的 base component alpha 以
normal source-over union 得到 `a_B(p)`，variant alpha 为 `a_V(p)`；两者都先映射到完整 canvas 并归一化到
`[0,1]`。每个 side/role 必须满足：

```text
mass_B = Σa_B(p)
mass_V = Σa_V(p)
coverage_leak = Σ[a_B(p) × (1 - a_V(p))] / max(mass_B, 1e-12) ≤ 0.01
alpha_mass_ratio = mass_V / max(mass_B, 1e-12) ≤ k_role
visible_contribution_i(p) = a_i(p) × Π[1-a_j(p), rank(i) < rank(j) ≤ rank(anchor)]
feature_scale_B = sqrt(mass_B)
base_context_radius = clamp(floor(c_role × feature_scale_B + 0.5), r_min_role, r_max_role)
authorized_i(p) ∈ {0,1}
unauthorized_i(p) = visible_contribution_i(p) × (1-authorized_i(p))
intrusion_i = Σ[a_V(p) × unauthorized_i(p)] / max(mass_V, 1e-12)
a_U(p) = Σ[unauthorized_i(p), i is an ordinary prefix part]
occlusion_intrusion = Σ[intrusion_i, i is an ordinary prefix part] ≤ 0.01
```

prefix 是从画布后方到 `draw_anchor_part_id`（含 anchor）的 ordinary draw sequence；anchor 之后的 part 会在
variant 之后重新绘制，不能被错误计入“会被遮挡”的集合。built-in blink/mouth roles 的 authorized underlay
始终包含 `base_part_ids`。所有 role 对 canonical `base_tag=face` 的唯一授权域都是
`dilate(support(a_B), base_context_radius)`；face 在该域外的 visible contribution 与其他 prefix part 一样
进入 `a_U`。授权域不能读取 `a_V`、variant bbox、variant mass 或 variant component geometry，因此不会在
被测积分域内退化成恒真条件。

形式上，`authorized_i(p)=1` 当且仅当 `i∈base_part_ids`，或 `base_tag(i)=face` 且
`p∈dilate(support(a_B),base_context_radius)`；其他 ordinary prefix part/pixel 均为 `0`。这条定义对
`support_preserving` 与 `support_expanding` 完全相同，两类 role 的差异只存在于 registry 冻结的
`c_role/r_min_role/r_max_role` 和 capability/authoring 语义，不能再分叉 scale estimator 或 authorization topology。

`support` 固定使用 `alpha≥1/255`，`dilate` 使用 Euclidean disk 并裁到方形 canvas；`floor(x+0.5)` 明确定义
non-negative round-half-up。授权 seed 以及报告/发布分桶使用的 `base_support_bbox/aspect_ratio` 都必须直接取
A-owned `MaskComponentPlan` 已清理并冻结的 base support，不能从 raw alpha halo 或 variant 重算；bbox/span
使用 half-open canvas 坐标，`aspect_ratio=max(width,height)/min(width,height)`。base 已保证非空，故两边都
必须为正；bbox/span/aspect ratio 只作诊断与 calibration strata，绝不能重新进入 radius 公式。disk、clamp 与
裁切都必须确定性实现。这里必须使用上述
source-over 可见贡献，不能把全部 amodal mask 做 raw union：后者会把眼睛下方本来就应被贴片覆盖的脸部底图
误判为侵入。反稀释由独立的 role-scale gate 负责；intrusion 的分母只使用 `mass_V`，因而始终是贴片自身
被非授权可见内容占用的真实分数。

eligibility/report 必须把每个 ordinary prefix part 的 `intrusion_i`（包括精确的 `0`）完整写成按 stable `part_id` 排序的
`intrusion_by_part=[{part_id,base_tag,ratio}]`；禁止只保留 top-N。另写
`intrusion_face_ratio=Σ[intrusion_i,base_tag(i)=face]` 与
`intrusion_other_parts_ratio=Σ[intrusion_i,base_tag(i)≠face]`，validator 必须复算并要求两者之和以及
`intrusion_by_part[].ratio` 之和都等于同一 `occlusion_intrusion` 结果。这样覆盖 nose/eyebrow 的失败不会显示成
“face intrusion 为 0，所以没有侵入”。

`coverage_leak` 防止闭眼线稿下仍透出睁眼瞳孔，`alpha_mass_ratio` 防止用巨大贴图稀释 intrusion，
`occlusion_intrusion` 防止 replacement patch 覆盖 role envelope 外的 face，以及 nose、mouth、eyebrow 等
未授权且在插入点实际可见的普通部件。旧的 bbox-distance `spill_mass` 可以作为 A report 的廉价
诊断/数据统计，但不再是 admission gate，也不得以任何半径替代 `a_U`。普通 base drawable 的 setup opacity 始终为
`1` 且不得出现在 native implementation 的 opacity target；variant setup opacity 为 `0`，只在对应 control
上升到所需端点时以 source-over 覆盖 base。任一门失败只使 native implementation 不 eligible，并按
native→procedural→unavailable 的既定选择继续；strict profile 若无替代实现再以 required capability 失败。
因此合格的闭眼/口型素材必须是覆盖至少 99% base alpha 的不透明 replacement patch，通常包含目标线条和
足够但仍落在 base-anchored role envelope 内的匹配肤色上下文；透明底睫毛线稿或覆盖到 envelope 外脸颊/
下巴的整块贴图都不是合法 native variant。

因为 A 在 B/C 之前就要完成 draw order 与精确 atlas dry-run，eligibility 不能推迟到 C。
`NativeVariantEligibilityPlan v1` 由 A 在普通 base masks/order 已冻结后运行，且与 profile 无关：

1. 先验证 manifest 语法/路径以及不依赖 component 的 base/anchor 引用；随后由 A 的 component-partition
   子步骤对全部普通 Part 和所有 manifest-valid variant candidate 构造一次 `MaskComponentPlan v1`。
   eligibility 只消费其中的稳定 labels、side 与 component IDs，再完成 role/base expected-set、coupled-side
   和 anchor 条件校验，不能自己再生成第二份 partition；
2. 计算 coverage/role-scale/occlusion-intrusion、side partition 和 role-level result，再按 primitive registry 的 expected role set 检查
   blink/mouth-form atomic bundle 完整性；
3. 先要求全部普通 see-through base regions 单独通过 TexturePagePlan；失败才是 item-level
   `texture_budget_exceeded`。再按冻结的 atomic group 顺序
   `blink.native < mouth_open.native < mouth_form.native` 逐组尝试；每次都对“全部 base + 已 admitted groups +
   当前 group”从空 plan 重跑同一 MaxRects，而不是向旧布局贪心追加；同时用 A 冻结、B 必须复用的
   component labels 计算正式 Live2D projected drawable count。能在四页内完整装下且 count `≤1001` 才
   admission；装不下记 `native_variant_texture_budget`，drawable 超限记
   `native_variant_drawable_budget`，都以整个 group warning/rejection 后继续下一组。不能拆 bundle、降采样、
   复用 draw-order 数值或阻塞 core item。这个 `<` 只表示 admission priority；
4. 只把**完整、quality-eligible 且 atlas-admitted bundle**中的 variant IDs 放入 `render_variant_ids`。每个
   rejected entry 仍以 `{variant_id,code,metrics,repair}` 进入 A cache/report，但不成为 Part、不参与 draw 或
   B mesh；
5. A cache 保存 canonical component-label digest、每个 role/bundle result、admission attempts、最终 exact
   TexturePagePlan input/order、`render_variant_ids` 和
   `native_variant_eligibility_sha256`。B 必须消费这些 labels 生成 mesh，不能重新 threshold 后得到另一 partition；
   C 用同一纯 validator 复算摘要并只从已物化 eligible bundles 选 native/procedural，禁止增删 render region。

A 内部的偏序因此固定为：普通 Part/payload 解析 → 普通 Part 的初始 draw order → 全候选
`MaskComponentPlan` → variant quality/atomic admission → final render-part set → anchor bundle expansion 与最终
`part_draw_rank` → final-admitted texture dry-run snapshot。实现可以共享纯函数或流式执行，但不能交换这些
有数据依赖的节点，也不能让 rejected variant 在 final rank/region 集合中留下占位项。

admission 使用正式 dual-runtime 的共同资源 envelope，即使本次运行 `spine_4_2_dev` 也不重新接纳一个只因
Live2D drawable 上限被拒绝的 variant；这是为了让同一输入的 A/B geometry、render-part/texture plan 与
symbol universe 不随开发 profile 漂移。C 中 required/optional format decisions 本来就由 profile 决定，允许
变化，不能把这句话误读为整个 `RigDocument` 逐字节相同。以后若产品确实需要最大化 Spine-only facial variants，应新增独立 profile/schema，而不是让 dev
模式悄悄改变公共 Rig。

所以“目录语法/路径/引用非法”是 `input_contract_mismatch` 并停止 item；“合法素材未过 coverage、组件或完整性
质量门”是稳定 capability rejection，可回退/omit。两者不能混成一个 catch-all。即使 rejected PNG 不进入
atlas，它仍进入 `native_variant_set_sha256` 和 stage input fingerprint，修改后会从 A 重算 eligibility。
目录缺失时 eligibility 使用对 `{schema_version:1,render_variant_ids:[],bundle_results:[]}` 做 JCS 的固定摘要；
它不是 set empty digest 的别名，plan/schema 变化仍必须改变 eligibility digest。

set digest 是对 schema/version、按 `variant_id` 排序的规范化 entry（含 PNG byte SHA）做 JCS 后的摘要；
manifest 的空白/key 顺序不改变语义摘要，PNG bytes、role/composite-mode/base-list/anchor/xyxy/path 任一变化都会改变。stage manifest
把这个 semantic set digest 作为输入，不再同时用 raw manifest whitespace 制造第二种失效语义。

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

`CanonicalTagRegistry v3` 同时拥有 source tag → Part internal ID 的唯一 codec。当前 23 个 base tag 都是
lowercase ASCII 单词、以一个空格分隔；`semantic_slug` 精确等于把每个空格换成 `-`，其他字符不变，
例如 `front hair → front-hair`。未分侧 Part ID 为 `part/<semantic_slug>`。只对上表允许 split 的 family，
source `-r` 映射为 `.xmin`、`-l` 映射为 `.xmax`；A 将一个未分侧 mask 的两个可靠 LR components 提升为
两个 side Parts 时也使用同一 `.xmin/.xmax` ID。`base_tag` 仍保存不带 source suffix 的 registry key（例如
`front hair`），source tag/layer 与 side provenance 另存；非 LR family 的多个连通分量继续属于同一 Part，
不能为了拿到好看的 ID 擅自拆成 side。任何 source tag、semantic slug 或 derived Part ID 重复/未注册都
`input_contract_mismatch`，不能用 display name、PSD 顺序或数字后缀消解。

`module/auto_rig/contracts.py` 读取时冻结这些细节：

- 所有必需 manifest/payload 与 `rig_inputs` 文件必须是 item root 下的 regular file，父目录链也不能含
  symlink/junction/reparse point；v1 拒绝所有链接而不是尝试证明某个链接“仍在目录内”；
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
owner stage；每个**成功提交的 payload** 只能出现在该 stage manifest 的 `output_file_sha256[]` 中；下游只能把它的摘要记录为
input/upstream dependency，不能改写文件后再把同一路径声明成自己的 output。唯一的结构性例外是
`rig/cache/<stage>/manifest.json` 本身：它是该 stage 最后写入的 commit marker，不把自己的 SHA 放进
自身 output 列表以避免循环摘要；其文件 SHA 由直接下游记录，terminal G 则由
`StageGraphValidator` 直接验证。`rig/cache/<stage>/failure.json` 是同一 owner 的**非成功 payload**：它只在
旧 commit marker 已撤销后存在，不进入成功 inventory，也永远不能用于 resume；G 只把其摘要当作失败输入。

| 文件 | owner | 契约 |
|---|---|---|
| `rig/cache/A/geometry_observations.json` | A | 私有 `GeometryObservationCache v1`；parts、cleaned masks/component labels、joint observations、variant admission/texture dry-run 与诊断证据 |
| `rig/cache/A/components/<sha256>.qcl` | A | 私有 `CanonicalLabelMap v1` blobs；B 只读，完整集合由 A manifest 精确列出 |
| `rig/cache/B/rig_geometry.json` | B | 私有 `RigGeometryCache v1`；canvas/parts/joints/bones/meshes/weights 与 A/B diagnostics，不是 `RigDocument` |
| `rig/cache/<stage>/failure.json` | 实际失败的 A-F stage | 非复用 `StageFailureRecord v1`；旧成功 commit 先失效，记录规范化诊断/异常分类/repair 与输入摘要，不得被当成 cache hit。A-E record 供 G 发布公共错误，F record 只属于实验评测 |
| `rig/cache/G/manifest.json` | G | 正式成功或任一 A-E item failure 的 terminal input/output digest、状态与当前可用 upstream manifest/artifact-set 摘要；G commit marker |
| `rig/cache/<stage>/manifest.json`、`rig/cache/<stage>/*` | 对应 stage | 私有缓存指纹、产物摘要、中间数组和调试图；路径不得跨 stage 共写 |
| `rig/rig.json` | C | 完整、公共、版本化 `RigDocument v1`；C 成功前不存在，D/E 只读 |
| `rig/report.json` | C | A-C 结构化 diagnostics、质量指标和全局符号表摘要；D/E 不追加，映射本体在 `rig.json` |
| `rig/motion_manifest.json` | C | `RigDocument.clips/expressions` 与全局符号表的确定性只读 projection；含逐格式状态、原因、文件引用和默认 clip 建议，不是第二事实源 |
| `rig/shared/textures/page_<index>.png` | C | canonical encoded PNG bytes；index 为从 0 开始、连续、无前导零的十进制；D/E 只允许 byte copy，不允许解码重编码 |
| `rig/export_manifest.json` | G | 成功终态；C/D/E manifest、G finalizer version、profile/validator 与全部发布 artifact-set 摘要 |
| `rig/error.json` | G | 失败终态；ordered failed stages/records、稳定 diagnostics、input/config/override fingerprint 与可执行修复建议 |
| `rig/spine/skeleton.json` | D | Spine 4.2 数据 |
| `rig/spine/skeleton.atlas` | D | Spine atlas |
| `rig/spine/textures/page_<index>.png` | D | 从 C canonical page 原样复制的 Spine atlas pages；SHA 必须相同 |
| `rig/spine/export_report.json` | D | global symbol table 的 Spine 子集/摘要、validator 结果和降级记录 |
| `rig/live2d/model.moc3` | E | MOC3 V4.00 二进制模型；v1 固定 basename，不从 item/display name 派生 |
| `rig/live2d/model.model3.json` | E | runtime 入口与资源引用 |
| `rig/live2d/model.cdi3.json` | E | 参数、部件的显示信息 |
| `rig/live2d/textures/page_<index>.png` | E | 从 C canonical page 原样复制的 Live2D 纹理页；SHA 必须相同 |
| `rig/live2d/motions/<preset>.motion3.json` | E | 固定动作 preset |
| `rig/live2d/expressions/<preset>.exp3.json` | E | 固定表情 preset |
| `rig/live2d/export_report.json` | E | global symbol table 的 Live2D 子集/摘要、参数映射、MOC consistency、runtime 验证和降级记录 |

这里的 `<index>` 精确使用 `TexturePagePlan` 的 0-based contiguous decimal index；`<preset>` 精确使用
`GlobalExportSymbolTable` 在对应 `motion` / `expression` namespace 中解析出的 `export_name`，再追加一次格式
扩展名。例如 `clip/wave.xmin` 的 v1 generic name 是 `wave_xmin`，不能直接拿 internal slug、display name 或
再次 sanitize 后生成文件名。Spine `animations{}` key 与公共 artifact fragment 同样使用其 `animation`
namespace export name。

逐字节确定性不能只靠“同一个 Python 进程通常保持 dict 顺序”。`CanonicalArtifactEncoding v1` 冻结：

- `DigestEncoding v1` 统一所有 auto-rig-owned JSON/manifest/report/API 字段（含 `NativeVariantManifest`）的
  SHA-256 表示：使用
  `sha256:` 加 64 个 lowercase ASCII hex；digest-derived 文件 basename 使用 64 lowercase hex 且不带前缀。
  大写 hex、base64、裸 JSON hex、截断摘要和混用前缀均由 schema 拒绝。既有 see-through upstream
  manifest 仍按自己的冻结 schema 解析，其整个合法 payload 参与 target digest，不能由 auto-rig 原地重写；
- 所有 auto-rig JSON（private manifest/cache descriptor、Rig/report/motion manifest、Spine JSON、
  model3/cdi3/motion3/exp3、G terminal artifact）使用 RFC 8785 JCS 的 UTF-8 bytes，无 BOM、无额外
  whitespace/尾随换行，输入必须满足 I-JSON：拒绝重复 object key、NaN/Infinity，并把 negative zero
  规范化；超过 IEEE-754 安全整数范围的计数/ID 必须按 schema 写十进制字符串，不能交给不同语言舍入；
- JCS 只解决 object key/string/number 编码，不替 schema 决定 array 顺序。part/mesh/control/clip/
  expression/symbol/diagnostic/file arrays 分别按其 canonical ID/order key；bones 按 parent topology，Spine
  slots 按 component draw rank，MotionClip keys 按 frame，纹理 page/region 按 page/index 与 stable part ID。
  validator 必须拒绝“先迭代 set/dict 再交给 JCS”的伪确定实现；
- `skeleton.atlas` 固定 printable ASCII、LF、page index 升序、每页 region 按 stable part ID、页间一个空行、
  文件末尾一个 LF；路径/export name 已由 symbol table 保证 ASCII。任何平台原生 CRLF 都非法；
- MOC3 的整数宽度、little-endian IEEE-754 float32 round-to-nearest-even、`-0→+0`、finite-only、逐 section
  packing/alignment/zero-fill 由 attested binary codec descriptor 冻结；禁止使用 native struct alignment，也禁止把
  `csmAlignofMoc=64` 的 runtime buffer-base 要求推广成所有 SOT section offset 的统一对齐规则。PNG 继续由
  `CanonicalPngEncoder` 单次生成；A 私有 component label 使用上文独立的 `CanonicalLabelMap v1`，不能拿
  pickle/NPZ 代替。JSON/atlas/QCL/MOC encoding profile/version 分别进入
  C/D/E/G fingerprint，不能把日志格式变化混进语义 encoder。

正式完成标记至少包含：

```json
{
  "schema_version": 1,
  "producer_stage": "G",
  "finalizer_version": "terminal-finalizer-v1",
  "input_fingerprint": "sha256:...",
  "native_variant_set_sha256": "sha256:...",
  "native_variant_eligibility_sha256": "sha256:...",
  "config_fingerprint": "sha256:...",
  "rig_overrides_sha256": "sha256:...",
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
  "motion_runtime_contract_sha256": "sha256:...",
  "global_symbol_table_sha256": "sha256:...",
  "texture_contract": {
    "schema_version": "shared-texture-v1",
    "canonical_uv_space": "page_top_left_v_down",
    "alpha_mode": "straight",
    "color_space": "srgb_bytes",
    "spine_uv_adapter": "spine-4.2-uv-v1",
    "spine_atlas_pma": false,
    "live2d_uv_adapter": "cubism-v4.00-uv-v1",
    "live2d_runtime_loader_contract": "sha256:..."
  },
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
列表必须与相应 D/E public output inventory 精确相等，不能只记录目录名、文件数量或忽略未声明的旧文件。
G manifest 再把上述 `export_manifest.json` 的最终 SHA 作为自己的 success
output，因此不在 export manifest 内反向记录 G manifest SHA，避免循环摘要。

公共失败终态至少包含：

```json
{
  "schema_version": 1,
  "producer_stage": "G",
  "finalizer_version": "terminal-finalizer-v1",
  "terminal_state": "failed",
  "item_id": "item/...",
  "target_input_fingerprint": null,
  "observed_input_set_sha256": "sha256:...",
  "native_variant_set_sha256": null,
  "native_variant_eligibility_sha256": null,
  "config_fingerprint": "sha256:...",
  "rig_overrides_sha256": "sha256:...",
  "profile": "dual_runtime_core_v1",
  "profile_fingerprint": "sha256:...",
  "validation_tier": "release",
  "failed_stages": ["D", "E"],
  "failure_records": [
    {"stage": "D", "sha256": "sha256:..."},
    {"stage": "E", "sha256": "sha256:..."}
  ],
  "failure_set_sha256": "sha256:...",
  "diagnostics": [],
  "retry_policy": "requires_input_config_or_code_change"
}
```

`ObservedInputInventory v1` 在任何契约 parser 之前构造，是 early-A failure 的 total evidence source。它只枚举
item 内固定授权候选：`src_img.png`、`layerdiff/manifest.json`、`optimized/**`、`final.psd`、
`final_depth.psd` 与 `rig_inputs/**`，明确排除 `rig/**`、root Marigold 内部 PNG/info 和日志。inventory 按
normalized relative path 排序，逐项记录 `missing/regular_file/directory/symlink/unreadable`；只对授权范围内
可读 regular file 取 byte SHA。它不跟随 symlink，symlink 只记录 link-text SHA，不公开或访问目标；目录
遍历也不穿过 reparse point。固定入口缺失要有 absence marker，不能得到与“从未检查该路径”相同的摘要。
若连 item root 都不能安全 `lstat`，batch result 使用 bootstrap code `item_discovery_failed` 并停止/跳过该
无法稳定寻址的候选（按 batch policy），不尝试伪造 per-item terminal，也不受 `continue_on_error` 的已枚举
item 语义保护。

成功时 canonical parser/A eligibility 已完成，`input_fingerprint`、`native_variant_set_sha256` 与
`native_variant_eligibility_sha256` 都必须是非空 digest；失败时
`error.json.target_input_fingerprint`、`native_variant_set_sha256`、`native_variant_eligibility_sha256` 分别在
对应 canonical set/plan 尚未形成时为 `null`，一旦形成则填真实值，绝不能用 raw inventory digest 冒充。
`observed_input_set_sha256` 对所有由 G 发布的 A-E item failure 必填，保证缺文件、坏 JSON 和路径攻击也能
产生稳定、可寻址的失败证据；F 的独立实验 failure 不伪装成 G terminal artifact。
`retry_policy` 不是异常处理器现场猜的布尔值，而是 `DiagnosticRegistry v1` 对稳定 code 的枚举投影；当前
可持久化 deterministic item failures 固定为 `requires_input_config_or_code_change`。若未来新增可在完全相同
fingerprint 下自动重试的 transient code，必须新增明确枚举与调度上限；G 写失败本身失败仍只走
`terminal_finalization_failed` batch result，不伪造这条公共 policy。

各阶段先使自己的旧 commit marker 失效并删除自己的陈旧 `failure.json`，再写 stage-local staging 目录并替换自己的产物；禁止 B 写
一个半成品 `rig.json` 再由 C 原地补字段。`output_file_sha256[]` 是 owner stage 的**精确产物集合**，
不是“至少列出这些文件”。C 的 `rig/shared/textures/`、D 的 `rig/spine/` 与 E 的 `rig/live2d/` 都是
可变 public output namespace：新 inventory 物化时必须删除该 owner namespace 中不再出现的旧 page、
motion、expression 或 report，validator 也必须拒绝任何未列出的 owner-owned public file。不能因旧文件
仍未被新 model/atlas 引用就把它留在交付目录。

C 从已验证的 A/B cache 与 C 自己的 binding/packing 结果组装完整文档，在 staging 中通过
`RigDocumentValidator` 和 projection/inventory validator 后，才逐文件原子替换
`rig.json`、`report.json`、`motion_manifest.json` 与 canonical texture pages；清除 obsolete C-owned 文件后，
把 C manifest 作为最后一个 commit marker 写入。D/E 使用同样的“旧 manifest 先失效、精确 inventory
发布、manifest 最后写”协议。D/E 只有在 C manifest 与全部 C-owned 文件摘要/精确 inventory 匹配后
才能启动；中途崩溃只会留下没有有效 commit marker 的 staging/partial output，不能成为可复用 Rig 或
正式 artifact set。

这里的 crash recovery 不以 rename 顺序或父目录 fsync 已成功为前提。每次 resume/skip 都必须重新打开并
byte-hash manifest 声明的全部输出，再扫描 owner public namespace；commit-marker-last 只减少可见撕裂窗口。
禁止用“mtime/size 未变”跳过 rehash，也禁止把旧进程内缓存的 `FileDigest` 当作磁盘事实。断电后即使 payload
与 marker 的目录项可见顺序异常，也必须因 digest/inventory 不一致而从最早失效 stage 重跑。

G 同时承担**正式成功终态**和**所有 A-E production-stage item failure 的公共发布**。正式 release profile 的 D/E fan-out
必须先全部 settle（成功、失败或明确取消）并收集 failure records，同一 item 的 G 恰好执行一次。
dev/structural profile 成功时不运行 G success path，以最后一个活动 stage 的 report 结束；它既不写 G
manifest，也不写 `export_manifest.json` 或 `error.json`。任一 profile 的已枚举 A-E item 失败仍调用 G failure
path：

1. success path 重新验证当前 C/D/E manifest 的 expected fingerprint、所有 output SHA、required
   capability、profile 与 validator 结果，构造完整 artifact-set digest，再写
   `export_manifest.json`；
2. failure path 接受一个或多个失败 stage 的规范化 `StageFailureRecord`，按 stage name 排序、形成
   failure-set digest 后写公开、稳定、per-item 可寻址的
   `error.json`；A/B/C/D/E 不能直接写这个路径；
3. G-owned `export_manifest.json` 与 `error.json` 在一次**已提交的 G terminal state** 中严格 XOR。success
   先删除旧 error，failure 先删除旧 export manifest；目标文件逐文件原子替换，G manifest 最后写入；
   success manifest 的 `output_file_sha256[]` 只列 export manifest，failure manifest 只列 error record；
4. job-startup gate 在 item 枚举前失败时仍没有 per-item `error.json`，只返回上一节定义的 job-level
   error；F 实验评测失败也不改变正式 A-E/G 终态；
5. 导出失败时允许保留 C 公共产物和 D/E 各自报告供重试，但不得为汇总错误去改写 C-owned
   `rig/report.json`。G 从 stage-local failure record 复制稳定诊断，CLI/job result 只负责批次聚合；
6. G 自己若因权限、磁盘或原子替换失败而无法发布 terminal artifact，必须移除无有效 manifest 的
   partial terminal file（best effort），在 batch result 返回 `terminal_finalization_failed`，且绝不能把
   item 记为 completed。这个错误不保证还能写进同一个 `error.json`；下次运行因 G manifest 缺失而重试。

`StageFailureRecord v1` 的 digest 只覆盖稳定的 stage/code/severity/entity/repair、规范化 exception class、
`observed_input_set_sha256`、可用的 canonical input/variant/eligibility digest 和配置摘要；时间戳、绝对路径、Python traceback、OS 本地化 strerror 和日志行不进入公共 record/digest，
只能留在 batch debug log。这样同一确定性失败可产生同一 G failure fingerprint，又不把调试细节伪装成 API。

`skip_completed` 只对正式 release profile 有定义；dev/structural job 使用普通 stage resume，传入
`--skip_completed` 必须在 item 枚举前拒绝，不能把 D-only 或 structural report 冒充产品完成标记。
正式路径的 `skip_completed` 不是 `Path.exists()`：它必须调用与正常 resume 相同的
`StageGraphValidator`，从当前
input/config/code/registry/profile 重新计算 A-E expected fingerprints，递归核对每份 stage manifest、
全部 output SHA 与精确 inventory；G 不接受调用方从旧 G manifest 回读再自比的 expected fingerprint，
而是从 `export_manifest.json` 重新计算 terminal config fingerprint，并核对其中的 C/D/E manifest、artifact-set、
motion/symbol/validator 摘要。只有 G manifest
状态为 `completed` 或 `completed_with_degradation`、其 output digest/inventory 匹配且整条 DAG 可复用时
才跳过 item。`export_manifest.json` 单独
存在、只有 `status=completed`、G cache 被删、D/E encoder/validator 改版或任一 artifact 被篡改，都
不得触发 skip；若只有 G cache 缺失而 A-E 全部有效，只重跑廉价 G，不重算 mesh/exports。
一旦 validator 判定 G 不可复用，调度器必须在启动最早失效 stage 前调用 G-owned
`invalidate_terminal()`，移除旧 G manifest、`export_manifest.json` 和 `error.json`；其他 stage 不能删除或
覆盖这些路径。任何 non-skipped dev/structural 重跑也先调用同一 invalidator 清除上一次 failure 留下的
`error.json`；这样开发成功后不会继续暴露陈旧失败，也不会在运行过程中继续暴露已知陈旧的 completed
标记。

状态不是自由文本，v1 冻结为三组互不混用的枚举：

- A-F success commit manifest：`stage_validated`、`stage_validated_with_degradation`；
- A-F 非复用 `failure.json`：`stage_failed`；失败记录永远不是可复用 commit manifest；
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
control/binding、motion/expression、format plans、candidate/symbol table 和 texture plan 组装为以下完整结构的阶段。

`RigGeometryCache v1` 为了让 C 在不重开 QCL/PNG 的前提下验证引用闭包，可以嵌入 A/B frozen plan
的规范化投影；这是一个有意的 v1 空间换确定性选择，不代表存在第二份可独立修改的业务事实。每个嵌套
plan 的摘要、component/Part、bone parent/joint、mesh topology/UV 和 influence/bone 引用都必须在读取时
重新校验。C 只消费这些规范化事实，不读取 B 的临时内存对象、A 的 QCL bytes 或源 PSD/PNG，也不得把
capability/control/clip/expression/format/symbol/texture-page 字段反写进 B cache。

首版至少包含：

```json
{
  "schema_version": 1,
  "generator": {"name": "qinglong-auto-rig", "algorithm_version": "1"},
  "input_fingerprint": "sha256:...",
  "input": {"tag_version": "v3", "coordinate_space": "layerdiff_canvas", "native_variant_set_sha256": "sha256:...", "native_variant_eligibility_sha256": "sha256:..."},
  "canvas": {"width": 1024, "height": 1024, "origin": "top_left", "y_axis": "down"},
  "parts": [],
  "joint_observations": [],
  "joints": [],
  "bones": [],
  "meshes": [],
  "capabilities": [],
  "control_specs": [],
  "control_bindings": [],
  "clips": [],
  "expressions": [],
  "format_plans": [],
  "primitive_candidates": {
    "schema_version": 1,
    "enumerator_version": "primitive-candidates-v1",
    "candidate_universe_sha256": "sha256:...",
    "items": []
  },
  "export_symbols": {
    "schema_version": 1,
    "namespace_schema_version": "export-namespaces-v1",
    "internal_id_codec_version": "internal-id-v1",
    "name_codec_version": "export-name-v1",
    "symbol_kind_codec_version": "symbol-kind-v1",
    "exporter_families": ["spine_4_2", "live2d_moc3_v4_00"],
    "symbol_universe_sha256": "sha256:...",
    "symbols": []
  },
  "texture_pages": [],
  "diagnostics": []
}
```

硬约束：

- 内部引用只使用稳定 ID，例如 `part/handwear.xmin`、`joint/elbow.xmin`；显示名称不参与引用。
  `InternalIdCodec v1` 固定 grammar 为 `<typed-prefix>/<slug>`，prefix 为 lowercase ASCII schema enum，slug
  匹配 `[a-z0-9][a-z0-9._-]{0,126}`。registry/canonical tag 已命名实体使用 registry 冻结的 semantic slug；
  schema-derived 匿名实体使用
  `<kind-token>_<64-lowercase-hex(SHA256(JCS(identity_record)))>`，其中 `identity_record` 必须含 kind-specific
  schema version 和完整 typed identity fields，不能含 display name、绝对路径、遍历序号或 export name。
  同一 internal ID 对应两份不同 identity record 是内部 invariant failure；输入提供了 v1 未注册实体 kind 时则
  `input_contract_mismatch`，不能清洗 Unicode display name 或临时截短 hash 生成 ID；
- `joint_observations` 保留 geometry、pose、override 的原始证据和各自分数；
- 最终 joint 有 `status: resolved | unresolved | overridden`、最终坐标、决策方法和证据 ID；
- bone 存 `{id, parent_id, head_joint_id, tail_joint_id, role}`，不只存一个 pivot；仅
  `role="synthetic_root"` 的 `bone/root` 允许 `parent/head/tail=null` 与 `length=0`，其他 bone 必须引用有效
  head/tail joints；
- mesh rest vertex 使用 canvas 坐标；mesh UV 使用部件纹理的 canonical top-left image space
  `[0,1]`（`u` 向右、`v_top` 向下），不是任一 runtime 的最终 UV；triangle 是扁平索引；
- 每顶点 `influences` 是 `[{bone_id, weight}]`，权重非负、和为 1，首版最多 4 个；
- `xmin/xmax` 表示图像空间事实；`anatomical_side` 单独可空，不能混成一个字段；
- 每个 part 必须保存由 `CanonicalTagRegistry` 解析出的 `base_tag`；split suffix、source layer name 与
  display name 分开保存，draw/capability policy 只匹配 `base_tag`；native variant 另保存
  `source_kind="native_variant"`、`variant_id/semantic_role/composite_mode/base_part_ids/draw_anchor_part_id`、
  `setup_visibility/draw_bundle_id` 与
  `composite_plan_id/coverage_leak/base_alpha_mass_u8_sum/variant_alpha_mass_u8_sum/alpha_mass_ratio/role_envelope_id/base_feature_scale_kind/base_support_bbox/base_support_aspect_ratio/base_context_radius_px/intrusion_face_ratio/intrusion_other_parts_ratio/intrusion_by_part/occlusion_intrusion/spill_mass_diagnostic/component_partition_digest`，并绑定
  `native_variant_set_sha256/native_variant_eligibility_sha256`，不能伪装成新 canonical tag 或 joint evidence；
- 每个 part 带 A 冻结的 canonical back-to-front `part_draw_rank`/draw-policy digest，每个 mesh component
  带 B 冻结的 `component_draw_rank`/expander digest；component rank 全局唯一且同 part 连续，D/E 只能做
  格式方向映射；
- 未识别和 merged 部件保留在 parts/diagnostics 中，不能静默归 root 后消失；
- capability 由已解析的 part/joint/bone/mesh 事实推导，preset 只能消费 capability，不能为满足
  preset 反过来伪造骨骼；
- `control_specs` 冻结 control domain/default/unit、standard/custom parameter identity 与 registry
  digest；clip/expression 只引用 control ID，不能在 exporter 内临时创建参数或改变 domain；
- `control_bindings` 是完整、profile-independent 的 control→canonical target 映射；每条带稳定
  `binding_group_id/implementation_id/implementation_kind/implementation_rank/implementation_bundle_digest/binding_id/control_id/target_id/property/transfer`；
  native overlay opacity binding 另带非空 `visibility_branch_id` 且 target 必须是
  `source_kind="native_variant"` 的 drawable；其他 binding（包括 procedural opacity）该字段必须为 `null`，
  普通 base opacity 不能出现在 native bundle；
  同一 implementation 内的 tuple 只能出现一次，同一 `implementation_id` 的记录构成不可拆 atomic bundle。
  clip/expression 只引用 control，由 C 的 format planner 在每个 semantic group 中恰选一个完整 bundle；
  v1 rank 固定 `canonical=0, native=10, procedural=100`，同 group 的 rank 必须唯一且不得用遍历顺序兜底。clip/expression 只能引用至少
  有一个合法 binding group 的 control，不能内嵌、覆盖或复制一份 transfer；
- clip/expression 使用稳定 control ID，不直接塞 Spine timeline 或 Live2D keyform 字段；同一个 preset 的
  语义、control curve 与 control binding/target transfer 只有一个事实源，但 optional preset 的
  `supported_formats` 可以是 required formats 的真子集，必须先存入 Rig，再由公共 motion manifest
  projection 明示；
- `format_plans` 保存每个 exporter 的静态模型 feasibility、完整 `FormatPresetSetPlan` 结果，以及逐 preset
  plan 所引用的 planner/input/output digest；required format 的静态/required-set plan 必须为 supported，D/E 只能复算核对，不能
  在 writer 中改变 drawable/section/texture 容量结论；
- `primitive_candidates` 是 C 阶段由完整 Rig/registry 一次性生成的 immutable candidate records；
  D/E 只能按 `candidate_id` 筛选，不能重新派生 typed primitive key；
- `export_symbols` 是 C 阶段冻结的完整 `GlobalExportSymbolTable`，包含 schema/version、symbol-universe
  摘要和 typed internal key 到 canonical namespace + ASCII export name 的一对一映射；D/E 只能读取
  子集，不能新增或改变 namespace/name；
- `texture_pages` 记录 page 尺寸、top-left pixel rect、padding/extrude、canonical `u/v_top` 映射和像素
  摘要；它是 C 阶段从
  parts 确定性派生的共享计划与 canonical byte payload，D/E 只能复制字节并翻译引用，不能各自重新
  装箱或编码。每页 canonical path 必须位于 `rig/shared/textures/`，同时携带
  `canonical_uv_space="page_top_left_v_down"`、`rgba_sha256/encoded_png_sha256/encoder_fingerprint`；指向
  cache、Spine 或 Live2D 输出目录均非法。D/E 的最终 UV 只能由各自版本化 `FormatUvAdapter` 从这份
  canonical mapping 生成，不能假设两个格式的 V 轴、atlas origin 或 half-texel 行为相同。
  每页还必须固定 `alpha_mode="straight"`、`color_space="srgb_bytes"` 与 texture-contract version；
  exporter 只能声明/适配 runtime loader，不能改写 pixel alpha mode。

`motion_manifest.json` 不是第二套动作事实源，也不是 exporter 可以独立修改的 capability 数据库。
它由版本化 `MotionManifestProjector v1` 从落盘后的
`RigDocument.control_specs/control_bindings/clips/expressions/capabilities/format_plans`、profile 和
`GlobalExportSymbolTable` 确定性生成，至少记录 `rig_json_sha256`、canonical
`motion_semantics_sha256`（覆盖 ControlSpec/ControlBinding/curves/transfers、preset descriptor、format decisions 与
runtime-application contract）、`global_symbol_table_sha256` 与 `projector_version`。C validator 必须从刚写入的
`rig.json` 重算 projection 并逐字段相等后才提交 C manifest。D/E 先读取 Rig 中的逐格式决策构造 binding
plan，再验证 public projection 与 Rig/符号表一致；projection 被手改、过期或与 Rig 冲突时
`input_contract_mismatch`，绝不能用它覆盖 Rig。`rig/report.json` 同样只是带 `rig_json_sha256` 的诊断
projection，不拥有新的结构事实。

`RigGeometryCacheValidator` 只验证 B-owned 几何/权重结构；`RigDocumentValidator` 则必须要求
`capabilities/control_specs/control_bindings/clips/expressions/format_plans/primitive_candidates/export_symbols/texture_pages`
九组 C-owned 字段存在并
满足各自 schema/digest/引用闭环，不能把空缺字段解释成“尚未运行 C”的合法状态。C 在保存
`rig.json` 后必须重新加载并通过完整 validator，防止内存模型和磁盘模型漂移；D/E 也先运行同一完整
validator，任何部分文档立即 `input_contract_mismatch`，不尝试从 cache 补齐。

#### 全局导出符号表

`GlobalExportSymbolTable v1` 在 C 阶段、任何 Spine/Live2D 剪枝之前生成。symbol universe 是以下
集合的并集，而不是某个 exporter 最终“存活节点”的集合：

1. 完整 `RigDocument` 的 part/joint/bone/mesh/control/control-binding/clip/expression internal IDs；
2. 版本化 preset、`ControlRegistry` 与 parameter registry 的全部公共 IDs；
3. 按 symbol-table schema 固定的 exporter family universe
   `{spine_4_2, live2d_moc3_v4_00}`，从完整 Rig 与完整 preset registry 推导的所有 typed derived keys，
   包括 Live2D `(bone_id, parameter_id)` rotation instances、warp/ArtMesh/Part，以及 Spine
   bone 和**逐 mesh component**的 slot/setup attachment（含 setup opacity 为 0 的合法 native-variant
   part components）；不得按
   本次 profile、required formats、capability 或 pruning 缩小集合；
4. 纹理 page/region 和 motion/expression artifact 的稳定 key。

因此 `spine_4_2_dev` 与正式 dual profile 对同一 Rig 也必须得到同一全局表；未来新增 exporter family
需要升级 symbol-table schema，而不是悄悄改变旧名称。

C 与 D/E 不各写一套派生规则。C 调用一次 `PrimitiveCandidateEnumerator`，以“不过滤 capability”的
模式遍历完整 `ControlBinding` 集合并生成 immutable `PrimitiveCandidateSet`；每条 record 至少包含
`candidate_id/typed_primitive_key/binding_template/required_rig_facts/driver_registry_entry_id`。candidate
set 写入 `RigDocument.primitive_candidates`，`GlobalExportSymbolTable` 直接消费其中的 typed keys。
“不过滤 capability”不表示伪造不存在的 format binding：`ControlSpec.format_bindings[format]=null` 时，
enumerator 不为该 control 生成该格式的 parameter/primitive candidate，并把这个确定性 absence 纳入
candidate-universe digest。比如 `wave.*` v1 只有 Spine candidates，Live2D selector 不可能请求一个后来
才发现不存在的 parameter key。

D/E 的 binding-plan builder API 只接受 candidate records 与 `RigDocument` 中已冻结的
clip/expression format decisions，返回所选
`candidate_id` 及模板实参；它们没有“从 bone/parameter 新建 typed key”的入口。所有 plan bindings 的
`candidate_id` 与 `primitive_target_id` 因而按构造属于 C 的 universe。D/E 若收到未知 candidate、发现
record 与 symbol table 摘要不一致，说明持久化数据或 registry/schema 漂移，必须
`missing_export_symbol` 失败，不能现场补 key/名称。运行期检查保留为 defense-in-depth，不代替 CI
的 superset 属性测试。

每个 Rig 源实体先得到与格式无关的 `base_export_name`。如果 Spine/Live2D 都对该实体做一对一表示，
两边共享这个 base，且在各自 namespace 无碰撞时直接使用同一裸名；某一真实 namespace 内发生碰撞时，
只有该 namespace 的碰撞类成员按下文 codec 改写 `export_name`，跨格式对照仍使用不变的 base + typed source，
不能为追求表面同名而把两个独立 namespace 合并。一对多派生实体则在同一 base 上追加由 typed derivation key 决定的稳定
suffix。例如 Spine 的 `bone/torso` 使用 torso base，而 Live2D
`(bone/torso, parameter/auto_idle)` rotation instance 使用同一 base 加 parameter suffix，不能因 Live2D 剪枝
后占用集合变小而换一个 base。

internal/derived key 不是靠拼接未转义字符串猜类型，而是 canonical typed record，例如
`{kind:"rotation_deformer", source_internal_ids:["bone/torso"], parameter_id:"parameter/auto_idle"}`。
每条 symbol 记录 `kind/source_internal_ids/namespace_key/base_export_name/export_name`。
`namespace_key` 同样是 canonical typed record，不能由点号或斜杠字符串拆分恢复语义。
其中 `skin_id/slot_id/directory` 等 scope 字段使用冻结的 internal ID 或 schema enum，不能使用尚待
消解的 export name，否则 namespace 分组和名称生成会形成循环依赖。

`InternalIdCodec v1` 与 `ExportNameCodec v1` 把“稳定名称”落实成唯一算法：

1. 每个 typed symbol candidate 明确记录一个 `base_source_internal_id` 和有序 `derivation_tokens[]`；前者必须
   是 `source_internal_ids[]` 中由该 kind schema 指定的主实体，后者每项匹配 `[a-z0-9][a-z0-9_]{0,31}`。
   exporter 禁止从已经拼好的字符串反向恢复这些字段；
2. `base_export_name` 取 `base_source_internal_id` 最后一个 `/` 之后的 slug，把 `.`/`-` 的每个 maximal run
   换成一个 `_`，其余字符因 `InternalIdCodec v1` 已保证为 lowercase ASCII alnum/分隔符；去掉首尾 `_`。
   空结果是 schema bug，不用 display name 兜底。匿名/tuple-derived source 必须先按上文
   `InternalIdCodec v1` 使用完整 64-hex identity digest 得到合法 internal ID；名称 codec 不再另造一套
   identity record 或截短 internal identity；
3. generic preferred name 为 `base_export_name`；有 derivation tokens 时为
   `base_export_name + "__" + join(tokens,"_")`。Live2D parameter namespace 中 registry 明确声明的
   standard/custom parameter export name 是唯一例外，按大小写原样保留并占用 reserved set；
4. preferred name 若长度 `≤63` ASCII bytes、在自己的 `ExportNamespaceKey` 内唯一且不碰 reserved name，
   就原样使用。若超长或同 namespace 有多个不同 typed key 得到同一 preferred name，则**碰撞类中的每个
   非 reserved member**都改成
   `truncate_at_ascii_boundary(preferred,46).rstrip("_") + "_" + first16(lowerhex(SHA256(JCS({namespace_key,typed_key}))))`；
   不能让遍历中“第一个”保留裸名。reserved parameter 自身不改，只有冲突的普通 candidate 加 suffix；
5. suffix 后仍重复、为空、非 ASCII 或 `≥64` bytes 时 `export_name_collision`，不靠加长 suffix、改变顺序或
   本 item 的存活集合继续试。codec/version、kind→base-source/derivation-token schema 和 reserved parameter set
   全部进入 C fingerprint 与 symbol-table digest。

`SymbolKindCodec v1` 进一步冻结 v1 symbol families，避免两个实现虽然都遵守上述 hash 公式，却为同一
typed key 选择不同 base 或 token：

| typed symbol family | `base_source_internal_id` | `derivation_tokens[]` / 规则 |
|---|---|---|
| Spine bone/skin、Live2D Part、part-level atlas region | 对应 bone/skin/part internal ID | `[]`；一对一实体直接使用 base |
| Spine slot/attachment-key/attachment-object、Live2D ArtMesh | 对应 component 的 owner part ID | part 仅一个 component 时 `[]`；否则为 `[component_token]` |
| Live2D RotationDeformer | pair 中的 bone ID | `["rot", id_token(parameter_id)]` |
| Live2D WarpDeformer | canonical target 的 owner part/bone ID | `["warp", id_token(control_id)]` |
| Live2D Parameter | parameter internal ID | 忽略 generic preferred name，使用 parameter registry 的大小写精确 reserved export name |
| Spine animation、Live2D motion/expression artifact | 对应 clip/expression internal ID | `[]`；format namespace 各自做 collision 检查 |
| texture page | `texture-page/page_<index>` | `[]`；index 规则仍由 TexturePagePlan 冻结 |

其中 `id_token(x)` 取 internal ID slug，再按 base 的 `.`/`-` run 规则归一化；registry 中会进入 token 的 slug
必须得到匹配 `[a-z0-9][a-z0-9_]{0,31}` 的结果，否则 startup 以对应 `invalid_*_registry` 失败。
`component_token = "c_" + first16(lowerhex(SHA256(JCS(component_internal_id))))`；这里 hash 的输入是完整
internal-ID **字符串的 JCS 值**，不是 component mask bytes、显示名或当前 exporter 的序号。多 component
attachment 因而会显式写 atlas `path` 指向 part-level region；单 component 时仍可依靠
`effective_region_path` 的默认同名规则。任何新 symbol family、base-source 选择、token 顺序或 single/multi-component
分支变化都必须升级 `symbol-kind-v1`，不能只改 exporter。

这里的 16 个 hex 是 64-bit 可读 suffix，不是假装数学上无碰撞；正确性来自最终完整 namespace uniqueness
validator，碰撞时宁可拒绝 item，也不临时发明第三种名字。单 component source 的 Spine attachment/atlas region 与
Live2D Part/ArtMesh 可以因 namespace 独立而共享裸 base；一对多 derived entity 则由显式 tokens 区分。

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
`parameter` namespace 中保留；普通项只在同一 namespace 碰撞时附加稳定 hash suffix。因而单 component
`topwear` 源实体派生的 Spine slot、attachment key、atlas region、Live2D Part 和 ArtMesh 可以都叫
`topwear`，而不会互相消耗名称；multi-component symbols 按上表添加 component token，同一 namespace 内
两个不同 typed key 仍必须消解或失败。

相同 typed key 在任何 profile、剪枝结果、MOC3/model3/cdi3、Spine JSON/atlas、motion/expression 和
报告中必须解析成同一个 `{namespace_key, export_name}`；共享 `source_internal_id` 的跨格式实体还必须
能通过 `base_export_name` 交叉对照。Spine exporter 必须计算
`effective_region_path = path ?? actual_attachment_name ?? attachment_key` 并断言它恰好命中一个
`atlas_region`；当 attachment 与 region 同名时允许省略 `path`，不因跨 namespace 的合法同名强迫每个
attachment 写冗余 path。

两份 exporter report 只记录各自使用的子集，但必须携带同一个 `global_symbol_table_digest`。MOC3 的
Parts、Deformers、ArtMeshes、Parameters `ids` 数组必须逐项来自该表，`.model3.json`、`.cdi3.json`
和 keyform bindings 只能引用这些已发射 ID。`motion_manifest.json` projection 的 artifact fragment 同样
通过该表生成；不得写 internal ID 后再由下游猜测清洗规则。缺 symbol、同一 namespace 内重复 export name、
超长、非 ASCII、引用不存在或同一 typed key 得到不同 namespace/name，均以
`missing_export_symbol` 或 `export_name_collision` 失败；不同 namespace 的相同裸名称不得误报碰撞。

### Resume 与失效规则

“文件存在即完成”被拒绝。每个 A-F success commit manifest 必须记录以下共同字段；G manifest 使用相同
输入/输出摘要骨架，但 `status` 必须取本节后面冻结的 terminal enum，并另外记录 terminal state 所需的
C/D/E artifact-set 或 failure-set 字段：

```text
stage_name
stage_schema_version
status = stage_validated | stage_validated_with_degradation
algorithm_version
upstream_manifests{stage_name: sha256}
input_file_sha256[]
target_input_fingerprint
native_variant_set_sha256
native_variant_eligibility_sha256
relevant_config_fingerprint
rig_overrides_sha256
output_file_sha256[]
output_inventory_sha256
```

`input_file_sha256[]` 不是把目录里所有原始字节无脑再 hash 一遍：每个逻辑输入必须在 manifest schema 中
恰有一种 invalidation identity。opaque PSD/PNG/model bytes 可直接列 byte SHA；已有明确 semantic digest 的
结构化集合只记录其 semantic 字段。特别是 NativeVariant 的 raw `manifest.json` 和逐 entry PNG 不再单独
进入 `input_file_sha256[]`，整个逻辑输入只由 `native_variant_set_sha256` 表示（PNG byte SHA 已包含在其 entry
内）；否则改 JSON 空白会形成与前文相反的第二套失效语义。raw manifest bytes 只允许出现在 failure-only
`ObservedInputInventory` 证据中。被 `target_input_fingerprint`、native set、override/config/model coherence
digest 或 `upstream_manifests` 已传递覆盖的文件同理不重复列；`input_file_sha256[]` 只容纳该 stage 额外消费、
尚无其他 identity 的 opaque bytes。validator 必须拒绝同一路径/逻辑集合同时以 raw 与 semantic mode 重复登记。

对 A-F，只有 status 为 `stage_validated*`、manifest 匹配、所有输出存在且摘要/结构校验通过才可复用；
G 只有 `completed*` 且整条 upstream DAG 同样有效时才可触发 completed skip，`failed` 终态从不复用为成功。
`status` 是 stage 执行后的结果字段，由 manifest/graph validator 单独校验，**不进入**调用方在执行前计算的
`stage_fingerprint`；否则 `stage_validated` 与 `stage_validated_with_degradation` 会让同一输入产生两个无法预先
推导的 expected fingerprint。degradation 仍进入 G 的 terminal payload/fingerprint，并决定最终
`completed` 或 `completed_with_degradation`，只是不能反向污染 A-E 的 resume key。
任一 A-F stage 失败时必须先移除自己的旧 success manifest，再以 `StageFailureRecord v1` schema 原子写
`rig/cache/<stage>/failure.json`；该文件不进入成功 manifest、不能命中 resume；A-E record 由 G 发布 error 后仍可删除重算，F record 不触发 G。
上游 stage、配置、算法版本
或 overrides 任一变化，从第一个受影响阶段向后失效。姿态 provider、模型 SHA-256 和预处理
版本必须进入 joints stage 指纹。`layerdiff/manifest.json`、`optimized/manifest.json`、
`optimized/info.json`、实际 payload、可选 NativeVariant manifest/PNG set 和 canonical-tag registry 版本都进入输入 stage 指纹；
`NativeVariantManifest` schema、role registry/role-envelope、composite coverage/scale/occlusion-intrusion、side/component selector、atomic admission priority 与 anchor-expansion
policy 同时进入 A/B/C fingerprint，修改时必须从 A 失效；不能只因 binding 最终在 C 物化就复用旧
coverage/component 结果；
`DrawOrderPolicy` version、depth quantization、semantic-precedence DAG 与 Kahn ready-set comparator
同时进入 A/C fingerprint；
`InternalIdCodec` schema/version 同时进入 A/B/C fingerprint；`mask-component-id-v1` 等 kind-specific
identity-record schema 从其最早 producer stage 起进入所有下游 fingerprint，修改 component identity 必须从 A
失效，不能只重建 C 的 symbol table；
`MaskComponentPlan` schema、threshold/connectivity/morphology/side classifier、label dependency/version 同时
进入 A/B/C fingerprint，修改时必须从 A 失效；
`MeshBuildPlan` schema、sampling/quantization/degeneracy policy、SciPy/Qhull/scikit-image version/options
与 `ComponentDrawOrderExpander` version 同时进入 B/C fingerprint；
`TexturePagePlan` schema/resource profile、padded-rectangle builder、MaxRects implementation/library version、
排序和 tie-break 设置必须**同时进入 A 与 C** fingerprint，因为 A 运行真实 dry-run、C 重算并物化同一
plan；只改其中任一项必须从 A 开始失效。preset/`ControlRegistry`/`ControlBinding` schema/parameter/primitive registry、
`RigidDriverRegistry` version/content、
`PrimitiveCandidateEnumerator` version、candidate-universe digest、global symbol-table/
`ExportNamespaceKey` schema、`ExportNameCodec`、`SymbolKindCodec` 及 reserved-name set 和
capability profile、`MotionManifestProjector`/report projector 与 canonical Rig JSON serializer version 进入
C stage 指纹；`CanonicalArtifactEncoding` 的 `DigestEncoding`/JCS profile 进入所有会写 JSON 的 stage，atlas text profile 进入 D，
MOC binary codec descriptor 进入 E；`RigTransformSemantics`/motion schema/`MotionRuntimeApplication` version
同时进入 C/D/E fingerprint，`SpineExpressionHold` version 只进入 C/D；
`SpineFormatPlanner`/`Live2DFormatPlanner` 的静态 `FormatModelPlan`、逐 preset 纯 preflight 与
`FormatPresetSetPlan` selection/conflict kernel
version/content 同时进入 C
及对应 D/E fingerprint，writer/layout/runtime validator 的其余实现只进入所属 exporter stage；
`CanonicalPngEncoder`/Pillow/zlib version、RGBA mode、压缩与 metadata
policy 只进入 C fingerprint，修改 encoder 只需从 C 开始失效；
Spine encoder、Live2D compiler、
MOC3 writer、Live2D artifact-layout schema、`SpineCoordinatePlan`/`SpineBindPlan`、`Spine42UvAdapter`/`CubismV400UvAdapter`、coordinate/UV schema
attestation、parameter map
和各 validator 版本进入对应 export
stage 指纹；G success fingerprint 包含 target/native-variant-set/native-variant-eligibility/config/override digests、terminal-finalizer version、profile/validation tier、motion runtime
application contract、shared texture
contract/两格式 UV adapter/Live2D target-loader contract、当前 C/D/E manifest SHA 和各自 artifact-set
digest，G failure fingerprint 则包含相同 finalizer/config 输入、已完成
上游 manifests、`StageFailureRecord` schema/normalizer version、有序 `failed_stages` 与规范化
failure-set digest。运行时线程数等不
影响结果的选项不进入指纹。

`relevant_config_fingerprint` 必须按 stage 的真实消费面投影，不能把整个 TOML 或执行 profile 无脑塞进每一
阶段。NativeVariant admission 已固定使用 dual-runtime 共同 envelope，所以只切换
`spine_4_2_dev` / `dual_runtime_core_v1` / `dual_runtime_avatar_v1` 时，A/B 继续复用，从 C 的
format/preset decision 开始失效；只切换 structural/release validation tier 时，C 也继续复用，只失效实际
执行目标 runtime gate 的 D/E 与 terminal G。未来 profile 若改变 A 的资源 envelope，必须升级对应
resource-profile/schema 并从 A 失效，不能借 `profile_fingerprint` 暗中改变同名 v1 语义。

manifest ownership 还必须满足集合不变量：任意两个 stage 的 `output_file_sha256[].path` 交集为空；
`output_inventory_sha256` 是按 canonical relative path 排序后的完整 `{path,size,sha256}` 列表摘要，且
stage-owned public namespace 的递归文件集合必须与该列表精确相等。
stage manifest 路径按 stage name 保留且不属于 payload output 集合；任何 stage 都不能覆盖另一 stage
的 manifest。
A manifest 只拥有 A cache，B manifest 只拥有 `RigGeometryCache` 等 B cache；C manifest 才拥有
`rig.json/report.json/motion_manifest.json` 与 canonical texture pages，G manifest 独占两个互斥的
terminal artifact。C 修改公共 Rig 不会改变 B output digest；相反，B cache 或 B manifest digest 变化会
通过 `upstream_manifests` 使 C 及 D/E/G 失效。D/E 各记录 C，G 同时记录 C/D/E；map key 按 stage
name 排序后参与 fingerprint。若只损坏
`rig.json` 中的 C-owned 字段，重跑边界从 C 开始，禁止为了重建公共文档重新计算 B 的 mesh/weights。
任何 manifest 声明了另一 stage 已拥有的路径都以 `stage_artifact_ownership_conflict` 失败，而不是“最后写入者胜出”。

### 部件规范化与左右状态

入口只构造一次 `CanonicalPartMap: tag → Part[]`。精确 tag 优先，再做受控前缀匹配；同一
canonical tag 的 bbox、mask union 和来源列表都保留，不再“第一个出现者胜出”。
NativeVariant adapter 在 see-through tag map 与 A eligibility plan 完成后，只把 `render_variant_ids` 以
`source_kind="native_variant"`、source `variant_id` 和 `part/native.<variant_id>` internal ID 加入 render-part
集合；rejected entries 只留诊断。variant
不能参加 joint observation、torso/head/limb mask union 或 pose crop，只继承已验证
draw anchor 的 `base_tag/depth_bucket/bone eligibility`。`base_part_ids` 只交给 NativeVariant coverage/component planner，
不能用来 union mask、平均深度或重算骨骼；这样替换素材不会反向移动自动关节。

四肢状态明确为：

- `split`：两侧 tag 完整；
- `partial`：只有一侧；
- `merged-separable`：merged mask 有两个可靠连通域；
- `merged-ambiguous`：粘连或组件质量不足；
- `missing`：部件不存在。

这里必须区分两个概念：上游 canonical tag 的 suffix registry 只回答输入是否可以合法出现 `-r/-l`；
A 的 geometry-side registry 回答一个**无 suffix** 的 cleaned mask 能否从两个可靠组件推导 image-side。
`MaskSideClassifier v2` 后者固定为 `V3_SPLIT_FAMILIES ∪ {legwear, footwear}`，因此 v3 中只能以 merged tag
出现的 `legwear/footwear` 仍可成为 `merged-separable`。它们若不是恰好两个通过面积占比门、且质心 x
可严格排序的组件，就必须保持 `merged-ambiguous`，不能因“通常有两条腿”强拆。

`merged-ambiguous` 首版刚性挂到 torso/root，并产生 error 级诊断；只有显式
`--allow-partial` 才允许继续双格式导出。深度用于 draw order、遮挡诊断和 merged 候选排序，
不在已经按语义限定候选骨骼后再重复“禁止后层接受前层骨骼权重”。

draw order 不是 exporter 自己排序。A 的 `DrawOrderPolicy v1` 从所有 final parts 一次性生成 canonical
back-to-front `part_draw_rank`。先把有限的 `depth_median` 按 item 内 `[min,max]` 归一化到 `[0,1]`，量化为
256 个 bucket；上游契约已冻结“值越小越靠前”，所以没有语义约束时 bucket 较大的远层先画。若
`max=min`，全部进入同一 bucket，不能除零或回退 PSD/dict 遍历顺序。

语义遮挡不是同 bucket 才生效的弱 tie-break。v1 registry 生成显式 behind→front DAG，记号
`a < b` 精确定义为“`a` 必须先画、位于 `b` 后方”。`HEAD_CORE_DRAWABLE_TAGS` 固定为
`{ears, face, eyewhite, irides, eyelash, eyebrow, nose, mouth}`；它是 draw-policy tag 集合，不是一个虚构
part ID。至少生成以下边：

```text
back hair < every(HEAD_CORE_DRAWABLE_TAGS) < front hair
ears < face
face < {eyewhite, nose, mouth, eyebrow}
eyewhite < irides < eyelash
ears < earwear
{face, eyewhite, irides, eyelash, eyebrow} < eyewear
```

registry 在加载时必须证明 DAG 无环。A 用 Kahn 拓扑排序；只在当前 zero-indegree ready set 内，按
`(depth_bucket desc, stable_part_id asc)` 选择下一个节点。因此 semantic edge 永远覆盖异常 depth，未被
边关联的部件仍由 depth/stable ID 决定，且不会构造一个可能非传递的自定义 comparator。若输入 tag
扩展后出现语义环，job startup 以 registry error 失败，不能删除一条边现场继续。

`headwear` 不在 v1 的 hard-edge registry 中是有意决定，不是漏项：帽檐与 front/back hair 的相对关系随角色
造型改变，没有一条跨样本恒真的方向。v1 对它继续使用 depth bucket 与 stable ID；未来若要覆盖歧义样本，
必须新增显式关系输入或版本化 override，再升级 `DrawOrderPolicy`，不能把某个样本的经验边写成全局规则。

DAG registry 的端点是 canonical `base_tag`，不是必须恰好存在的 payload key。A 先把每个端点展开为
当前 final Part 中 `part.base_tag` 相等的稳定 ID 集合，再对一条 `a<b` 生成两集合的笛卡尔积边；因此
`eyewhite-r/-l`、`irides-r/-l` 等 split part 仍继承 base semantic order。任一端点集合为空时该规则对
本 item 不产生节点或边，禁止创建 phantom part；suffix、PSD layer name 和 alias display name 都不能绕过
`base_tag` registry。

NativeVariant 不把 `base_part_ids` 全部折叠或搬到一起。A 先排除 variants，对普通 see-through Part 完成
上述 DAG/depth 拓扑，得到 `ordinary_part_order` 与从 0 开始的临时 `ordinary_draw_index`；随后验证每项
`draw_anchor_part_id` 是其 base 列表中 index 最大、也就是最前的一层。按 anchor 分组后只把该 anchor token
扩成 atomic draw bundle：顺序为 anchor、再按 `(semantic_role,variant_id)` 排序的 variants；其他普通 base 的
**相对顺序**保持不变。扩展完成后才把整个 final sequence 重新编号为从 0 开始、全局唯一且无空洞的
`part_draw_rank`。所以 anchor 后方的普通 Part 数值 rank 可以整体平移，不能要求插入 variant 后还保持旧
整数；任何外部 edge 都连接 bundle 边界，不能插到 anchor 与 variants 之间。这样多层 eye base 可共享
occlusion coverage set，却不会为了闭眼素材破坏 `eyewhite < irides < eyelash` 等原始遮挡。

`ordinary_part_order`、最终 `part_draw_rank` 与 policy/depth-input/anchor-expansion digest 写入 A cache。B 生成 mesh components 后，由
`ComponentDrawOrderExpander v1` 按 `(part_draw_rank, stable_component_id)` 为每个 component 分配从 `0`
开始、全局唯一且无空洞、同 part 连续的 `component_draw_rank`，写入 B cache；C 只把两级 rank 组装进
最终 Rig。policy version/DAG/ready-set comparator 同时进入 A/C fingerprint，expander version 进入 B/C
fingerprint。D 把升序 component rank 映射为 Spine back-to-front slot order；E 只在 C 的 Live2D
`FormatModelPlan` 已证明 component count `≤1001` 后，把 rank 原值写入 Cubism `0..1000` draw order；
两边禁止重新读取 `depth_median`、PSD layer order 或 tag 名排序。v1 所有 part 使用 normal source-over blend，
preset 不生成 draw-order timeline；以后要换前后手必须作为新 MotionClip property/schema，而不是 exporter
临时交换 slot。

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

1. target-bound override 先通过 `observation-anatomy-validator-v1`；在 anatomy support 外的坐标只有显式
   `allow_outside=true` 才合法，合法后无条件优先，且授权 ID 必须进入 warning set；
2. geometry 高置信度时采用 geometry；
3. geometry 低置信度、pose 通过自身阈值和 mask/骨长校验时，采用 pose 并投影到合法区域；
4. 两者都有效但差异超过局部肢体宽度阈值时，标记 `unresolved/pose_disagreement`；
5. 部件不存在时 geometry/pose 不创建相关 joint/bone；存在但求解失败时保留 unresolved，绝不填画布中心。
   唯一例外是通过 target/anatomy gate 的显式 override，它可以补齐缺失关节，但不能由 resolver 自己猜出。

固定“膨胀 8px”也与分辨率绑定。`observation-anatomy-validator-v1` 固定使用
`min(max(2px, 1.0 × local_limb_radius), 0.05 × canvas_edge)`；没有局部半径时只给 2px 数值余量，
没有任何可用 evidence mask 时视为在 support 外。该门同时用于 pose 与 override，不能由 provider 名称绕过。

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
root(parent=null): synthetic identity frame, head/tail joint=null（永远存在）
lower_torso(parent=root): pelvis → spine
torso(parent=lower_torso): spine → neck
neck(parent=torso): neck → head_base
head(parent=neck): head_base → head_top
upper_arm.{side}(parent=torso): shoulder → elbow
forearm.{side}(parent=upper_arm.{side}): elbow → wrist
hand.{side}(parent=forearm.{side}): wrist → hand_tip          （有可靠端点才创建）
thigh.{side}(parent=lower_torso): hip → knee
shin.{side}(parent=thigh.{side}): knee → ankle
foot.{side}(parent=shin.{side}): ankle → toe                  （有可靠端点才创建）
```

创建顺序由 parent 拓扑排序得到，诊断连线由 head/tail 得到。骨骼的 `requires` 只引用
canonical part/joint 状态。`bone/root` 是唯一例外：它不是解剖骨，不消费 observation，rest transform
固定为 canvas identity、length=0，始终存在。其他父骨不存在时允许上提到最近存在祖先并最终落到
`bone/root`，但自身 head/tail joint 缺失时仍不创建。这样既不会生成 pivot 为 `(0,0)` 的假解剖骨，
也不会让半身/头像失去格式级共同根。

### 图层归属、网格与权重

part 先根据语义得到允许影响它的 bone 集合，再计算权重：头发、脸和衣服不会因为空间距离
接近手臂就被手臂骨骼吸走。`tail/wings/objects` 首版刚性绑定 root/torso 并标记
`dynamic_candidate=true`，不伪造关节链。

mask partition 与 mesh topology 分属两个 owner。A 的 `MaskComponentPlan v1` 对每个普通 part 和每个已通过
manifest/path/reference 前置校验、但尚未经过 coverage/resource admission 的 NativeVariant candidate 执行且只执行一次：固定 alpha threshold、4/8-connectivity、
孔洞/小组件阈值和 morphology；按 cleaned component mask 的 canonical `(bbox.y,bbox.x,mask_sha256)` 排序，
并把
`{schema:"mask-component-id-v1",part_id,component_bbox,cleaned_binary_mask_sha256}` 作为完整 identity record，
按 `InternalIdCodec v1` 派生
`component/c_<64-lowercase-hex(SHA256(JCS(identity_record)))>`；不能使用库返回序号、label number 或截短
digest。A 同时冻结 xmin/xmax/none side classification、label image/component-mask SHA、base projected
component count 与 plan digest。variant admission/drawable guard、B mesh、base-support coverage 和 C model plan
都消费这份集合；任何下游不得重新 threshold、merge 或 split。

A 不能只保存 label digest 后让 B 从源 alpha 重算。每个 part 的 tight crop label image 必须物化为私有
`CanonicalLabelMap v1`：路径固定为
`rig/cache/A/components/<full-label-map-sha256>.qcl`，binary layout 是 4-byte ASCII magic `QCL1`、
little-endian `uint32 width`、`uint32 height`，随后恰好 `width×height` 个 row-major little-endian `uint32`
labels；`0` 是背景，`1..N` 严格对应 canonical component record order，无 padding/trailing bytes。JSON record
另存 canvas crop `xyxy`、每个 label→component ID、blob path/size/SHA。文件名使用完整 64 个 lowercase hex
字符、不带 `sha256:` 前缀；读取到
同名不同 bytes 时先把 A commit 判为不可复用并从 A 重算，若 fresh staging 仍试图为同一 digest 产生不同 bytes，
则按内部 hash/codec invariant 失败，绝不能“最后写入者胜出”。codec/version 进入 A/B/C fingerprint 与 A output inventory；B 只读
并核对 magic/尺寸/labels/摘要，禁止调用 threshold/connected-components。公开 Rig 不引用 cache path，只保存
component IDs、mask SHA 和派生 mesh。

B 的网格步骤从 A-owned component masks 开始：

1. 每个冻结连通分量分别 contour/resample/interior sampling；
2. 空间哈希去重；
3. 每分量 Delaunay；
4. 通过重心、边多点采样和相对外接圆阈值删除跨透明区/狭长三角形；
5. 生成稳定 vertex order、flat triangles、boundary/hull 顺序和局部 UV。

“稳定”不能只指保存时排序。`MeshBuildPlan v1` 把会改变拓扑的 B 步骤一并冻结：

- contour winding 和起点选择均版本化；contour 起点取 canonical
  `(y,x,source_sample_id)` 最小项，不能使用 OpenCV/scikit-image 返回的首项；
- boundary/interior sample 不读全局 RNG。若算法需要抖动，seed 由
  `SHA256(part_id, cleaned_mask_sha256, mesh_plan_version)` 派生；持久化 vertex 在 triangulation 前量化到
  `1/256 canvas px`，相同量化点按 stable sample ID 去重；
- Delaunay 输入按 canonical vertex key 排序。禁止 Qhull 的随机 `QJ` joggle；共圆/近共线退化只允许在
  **拓扑副本**上施加由 stable vertex rank 决定、幅度 `<1/4096 px` 的 deterministic symbolic perturbation，
  输出坐标仍用未扰动的量化点。无法得到所有面积大于 epsilon 的合法拓扑时返回 `degenerate_mesh`；
- 最终 vertex order 固定为 hull/boundary 在前、interior 在后，各自按 schema 的 winding/start/key；每个
  triangle 先按 canvas signed-area 统一 winding，再循环移位到最小 vertex index 在首位，最后按 index tuple
  字典序排序。不能把 Qhull simplex 顺序写进 Rig；
- `MaskComponentPlan` schema/morphology/scikit-image component-label dependency 进入 A/B/C fingerprint；
  `MeshBuildPlan` schema、SciPy/Qhull/scikit-image contour dependency、Qhull options、量化/epsilon 与采样器版本进入 B/C
  fingerprint 和对应 report。byte determinism 只对该锁定环境矩阵承诺；升级依赖必须重跑 partition/mesh golden，
  不能靠 JCS 把不同拓扑包装成“同语义”。

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
继续留成 E 阶段的后置实施问题。

`TexturePagePlan v1` 冻结为：

- `2048×2048` 方形 RGBA page，最多 `4` 页；不自动降采样，不自动切换 4096；
- used page index 固定为 `0..N-1`，文件名片段精确使用无前导零 ASCII decimal：`page_0.png` 至
  `page_{N-1}.png`。C/D/E 三个 texture 目录必须复用同一 basename；1-based、零填充或稀疏 index 非法；
- **装箱单位固定为 canonical render-part payload，不是 mesh 连通分量或 Live2D ArtMesh**。每个
  `optimized/info.json.parts[tag]` 以及 A `render_variant_ids` 中每个 admitted NativeVariant 恰好生成一个
  region；合法但 rejected 的 variant candidate 不生成 region。源矩形就是该
  payload 的完整 `xyxy` crop；两类 ID 在统一 render-part namespace 中不得碰撞；
  透明空洞仍占 atlas 面积。B 阶段产生的多个连通分量，以及未来通过 Glue gate 后可能产生的
  skinning 子网格，都引用同一个 part region，并从原 canvas 坐标推导各自 UV；实验 E0-S 也不得
  借重新裁纹理掩盖几何缝；
- 每 region 的几何分成三层：原始 part crop 是 `content_rect(w,h)`；向外 `2px` 是复制 content 边缘
  RGBA 的 `extrusion_ring`；再向外 `2px` 是保持透明的 `safety_gap`。因此 MaxRects 使用的
  `packed_footprint` 精确为 `(w + 2×(2+2)) × (h + 2×(2+2)) = (w+8)×(h+8)`，`rotation=false`。
  `padding=2` 专指 safety gap，不能把 extrusion 算进 padding 后得到 `w+4`；
- 使用成熟的 MaxRects BSSF 实现，依赖版本进入 lock/fingerprint；输入先按
  `(max_side desc, area desc, stable_part_id asc)` 排序，所有同分选择再以 page/y/x 排序；
- 任一 mandatory see-through base region 超过可用单页尺寸，或完整 base set 四页装不下时返回
  `texture_budget_exceeded`，不得退回一部件一页；native candidate 超页/使当前 admission plan 超页则按上节
  原子拒绝该 group并记 `native_variant_texture_budget`，不能把 optional 资源失败升级成 base-set error；
- A 的同一 admission attempt 还按冻结 component labels 检查 projected Live2D drawable count；candidate group
  使总数超过 1001 时以 `native_variant_drawable_budget` 原子拒绝。mandatory base 自身超限仍由 C-owned
  `FormatModelPlan` 以 `draw_order_capacity_exceeded` hard fail，A 的预检不能冒充最终格式结论；
- C 按 plan 合成 row-major RGBA pixels，并通过单一 `CanonicalPngEncoder v1` **每页只编码一次**到
  `rig/shared/textures/page_<index>.png`。v1 输入固定为 C-contiguous straight-alpha `uint8 RGBA`，使用项目锁定的
  Pillow + zlib runtime、`optimize=false`、`compress_level=9`，不传入 PNGInfo/ICC/EXIF/DPI，也禁止
  `tIME/gAMA` 等非确定或改变色彩解释的 metadata；Pillow/zlib 版本、RGBA/alpha/color-space contract、
  参数和 metadata policy 全部进入 C
  fingerprint。不承诺跨未锁定 encoder/runtime 版本得到相同字节；
- page、`packed_footprint/extrusion_rect/content_rect` 的 top-left pixel rect、canonical `u/v_top`、part
  ID、`rgba_sha256`、`encoded_png_sha256` 和
  canonical relative path 写入
  `RigDocument.texture_pages`；两个摘要分别验证像素语义和实际交付字节，不能混成一个含糊的
  “page SHA”；
- D/E 只能以 byte-stream copy 从 canonical path 写各自 page，禁止调用 image decoder/encoder、改变
  metadata 或重新 pack。两份目标文件的 SHA 都必须等于 `encoded_png_sha256`，所以跨 exporter
  逐字节相同是 construction invariant，不依赖两个编码器碰巧采用相同 zlib/filter/chunk 策略。Spine
  atlas 与 MOC3 texture index 只翻译同一份 region plan；最终 UV 由 D/E 各自的 adapter 从 canonical
  `page_top_left_v_down` 映射生成，不属于 C 的共享 byte identity。

canonical pixel contract 另固定为 `RGBA8 + sRGB byte semantics + straight alpha`。C 不预乘 RGB；
extrusion ring 逐像素复制 content rect 最近边缘的 straight RGBA（含 alpha），safety gap 保持
`RGBA(0,0,0,0)` 后再编码。canonical UV 与 Spine atlas/MOC region 均只指 content rect，绝不能把 ring
或 gap 当成可见纹理；`RigDocument.texture_pages` 每页写
`alpha_mode="straight"`、`color_space="srgb_bytes"` 和 contract version。PNG 不带 ICC/gAMA 等可改变
解释的 profile metadata；这不等于可以省略 runtime 的 alpha 配置。D 的 atlas page 必须显式写
`pma:false`。E 的标准 model3 没有自定义 PMA 开关，因此 `export_report.json` 与公共 manifest 必须记录
`texture_runtime_contract`：Native 对未预乘像素使用
`CubismRenderer::IsPremultipliedAlpha(false)`；Web 因 renderer 采用 premultiplied workflow，按官方示例在
upload 时设置 `UNPACK_PREMULTIPLY_ALPHA_WEBGL=1`。任何目标 runtime 若采用不同 loader 约定，必须在 release matrix 单独
验证，不能改 canonical PNG 后仍声称跨格式 byte identity。

A 阶段先对 mandatory base，再对每个 native admission attempt 与最终 accepted set，用上面 `(w+8)×(h+8)`
packed footprint 运行与 C **同一实现、同一排序**的 MaxRects dry-run。每个 plan snapshot 都写
`sum_padded_region_area = Σ packed_footprint_width × packed_footprint_height` 和两个不可混用的比率：

```text
budget_occupancy = sum_padded_region_area / (4 × 2048 × 2048)
used_page_fill   = sum_padded_region_area / (used_page_count × 2048 × 2048)
```

前者衡量固定四页预算消耗，后者衡量实际已用页的装填效率；字段路径明确区分
`mandatory_base`、`admission_attempts[group_id]` 与 `final_admitted`，禁止再输出含糊的
`sum_padded_bbox_area / total_page_area`。报告还包含 used page count、最大 region 和失败/拒绝原因。
这不是只看总面积的估算：总面积小于容量仍可能因形状装不下。只有 mandatory base plan 失败才停止
item；native attempt 失败只拒绝该 atomic group。C 从 A 的 final-admitted part-region 输入摘要重算并要求
plan 逐字段一致，再写 pixels/UV。B/E 不得因 mesh 拆分改变 region
集合。D/E 不生成 pixels，只校验并复制 C-owned canonical PNG。v1 分辨率白名单下“单 part 超页”分支通常不可达，但作为 schema/未来 profile 的防御检查
保留；真正会触发的主要是四页组合装不下。

4×2048² 是 v1 的产品资源预算，不是假装成 Cubism 格式极限。以后提高页数/尺寸必须作为
profile/version 变更，并在目标 Native/Web/Unity runtime 上重新测峰值纹理内存与加载时间。

### 人工校正

首版不做画布编辑器。`rig_overrides.json` 也是版本化输入：

```json
{
  "schema_version": 1,
  "target_input_fingerprint": "sha256:...",
  "joints": {
    "joint/elbow.xmin": {"x": 812, "y": 1043}
  },
  "tag_aliases": {
    "objects_2": "handwear-r"
  }
}
```

坐标使用 Rig canvas 空间。`target_input_fingerprint` 是对 see-through manifests、合法 PartSource payload、
canvas metadata 与 canonical-tag schema 做 canonical 排序后的摘要；它**不包含** override 文件自身、
auto-rig 算法版本、运行配置或可选 native variant set，避免循环摘要，也允许同一 base canvas 在调阈值/
替换表情素材后继续复用人工关节坐标。variant manifest/PNG 另形成
`native_variant_set_sha256`；目录缺失时固定为 canonical empty-set digest，不用 `null`/空字符串。该字段与
由 A 产生的 `native_variant_eligibility_sha256`、stage manifest 的 `rig_overrides_sha256` 是彼此独立的值，
任何实现不得把其中一个代替另一个。set digest 说明“提供了什么”，eligibility digest 说明“哪些完整 bundle
通过 v1 质量门并进入 render set”，两者不能合并。
target fingerprint 不匹配默认拒绝应用，防止把 A 图的 override 套到 B 图。
`RigDocument.input_fingerprint` 与成功 `export_manifest.input_fingerprint` 使用同一个 target-input 定义；
失败 terminal 改用 nullable `target_input_fingerprint` 加必填 `observed_input_set_sha256`，因为 early-A 错误可能
根本无法形成 canonical target。native variants、config 与 override 摘要始终放在各自字段，不能悄悄并入后还保留同名字段。
override 后分两层复核：A 的 `observation-anatomy-validator-v1` 先做 mask-support 距离校验；B 在
`BoneSpec` 拓扑和相邻 resolved joint 已形成后再做骨长/零长度校验。确需放在 mask 外必须显式
`allow_outside=true` 并写 warning；该标志只授权 mask 外坐标，不跳过 B 的骨长门。
文件摘要变化使 joints 及之后阶段失效。

NativeVariant 的 role/base/draw 关系只来自上一节的独立 manifest，不放进 override 形成第二套映射。
`base_part_ids` 定义 occlusion coverage support，不能被重排成一个假想连续组；A 只把
`draw_anchor_part_id` 与共享该 anchor 的 variants 作为 atomic draw bundle，固定 anchor 在前、variants 按
semantic role/variant ID 紧随其后并分配连续 `part_draw_rank`。variant 保留自己的纹理/mesh/ID，只继承
anchor 的 draw/bone eligibility；D/E 不按名称重建关系。
override 也不能强制接纳 coverage/role-scale/occlusion-intrusion/component/resource gate 已拒绝的 variant、修改 admission priority 或
提高 atlas/drawable budget；这些都是版本化产品契约，不是人工关节校正。

`rig_overrides_sha256` 是 total `OverrideInputIdentity v1`，不是缺文件时随手写 `null`：文件不存在时取
`SHA256(JCS({schema:"rig-overrides-input-v1",present:false}))`；存在时先确认它是 item root 下非链接 regular
file，再取 raw byte SHA，最后摘要
`{schema:"rig-overrides-input-v1",present:true,file_sha256:<sha>}` 的 JCS。raw whitespace 变化因此会明确
失效 A/joints，这是对人工审计文件的保守选择；不能同时再把 override path 放进 `input_file_sha256[]`。
JSON/schema 无效时 identity 仍可形成并进入 failure record，但内容不应用。

### 批量动作与表情 preset

“一键”在这里指无交互批处理：每个 item 从 `RigDocument` 推导 capability，套用版本化 preset，
直接写入目标包。它不启动播放器、不截预览、不渲染 GIF/WebM。动作生成和格式编码必须分开：
`PresetBinder` 只从 C 已冻结的 `ControlSpec` 与 `ControlBinding` 产生 exporter-neutral `MotionClip`、
`ControlCurve` 和 `ExpressionPreset`；`TargetTransfer` 只存在于 binding，Spine 与 Live2D exporter 只能翻译这些记录，不能各自重新猜
动作、参数、表情或关节。

```json
{
  "id": "clip/idle",
  "preset_version": "motion-core-v1",
  "sample_rate_hz": 30,
  "duration_frames": 120,
  "interpolation": "linear",
  "loop": true,
  "control_curves": [
    {
      "control_id": "control/idle",
      "keys": [[0, 0.0], [30, 1.0], [60, 0.0], [90, -1.0], [120, 0.0]]
    }
  ]
}
```

与该 clip 分离、在 Rig 顶层只保存一次的 binding 是：

```json
{
  "binding_group_id": "binding-group/idle.torso",
  "implementation_id": "binding-impl/idle.torso.rigid-v1",
  "implementation_kind": "canonical",
  "implementation_rank": 0,
  "implementation_bundle_digest": "sha256:...",
  "binding_id": "binding/idle.torso.rotation",
  "control_id": "control/idle",
  "target_id": "bone/torso",
  "property": "rotation",
  "visibility_branch_id": null,
  "transfer": {"kind": "affine_scalar", "output_at_default": 0.0, "gain": 1.0}
}
```

控制值与目标属性值是两层数据，不能再共用 `channel.keys`，也不能把 transfer 复制到每个 preset。
`ControlSpec v1` 由 C 从版本化
`ControlRegistry` 物化到 `RigDocument.control_specs`，至少含
`{control_id, min, default, max, unit, format_bindings, registry_digest}`；其中
`format_bindings.live2d_moc3_v4_00={parameter_id, standard_id}`；只被 Spine-only optional preset 使用的
control 可显式为 `null`，但此时任何引用它的 Live2D preset plan 都必须 omitted。这样避免把 Live2D
字段伪装成 control 的跨格式身份。数值要求有限、
`min < max` 且 `min ≤ default ≤ max`。首版核心 registry 冻结为：

| control | parameter internal ID | Live2D export name | `min/default/max` | control unit |
|---|---|---|---|---|
| `control/body_sway` | `parameter/body_angle_x` | `ParamBodyAngleX` | `[-10, 0, 10]` | degree |
| `control/idle` | `parameter/auto_idle` | custom `ParamAutoIdle` | `[-1, 0, 1]` | normalized |
| `control/head_shake` | `parameter/angle_x` | `ParamAngleX` | `[-30, 0, 30]` | degree |
| `control/head_nod` | `parameter/angle_y` | `ParamAngleY` | `[-30, 0, 30]` | degree |
| `control/breath` | `parameter/breath` | `ParamBreath` | `[0, 0, 1]` | normalized |
| `control/eye_open.xmin` / `control/eye_open.xmax` | `parameter/eye_open.xmin` / `parameter/eye_open.xmax` | custom `ParamEyeOpenXMin` / `ParamEyeOpenXMax` | `[0, 1, 1]` | normalized |
| `control/brow_y.xmin` / `control/brow_y.xmax` | `parameter/brow_y.xmin` / `parameter/brow_y.xmax` | custom `ParamBrowYXMin` / `ParamBrowYXMax` | `[-1, 0, 1]` | normalized |
| `control/mouth_open` | `parameter/mouth_open_y` | `ParamMouthOpenY` | `[0, 0, 1]` | normalized |
| `control/mouth_form` | `parameter/mouth_form` | `ParamMouthForm` | `[-1, 0, 1]` | normalized |
| `control/wave_lift.xmin` / `control/wave_lift.xmax` | `null` | `null`（Spine-only v1） | `[0, 0, 1]` | normalized |
| `control/wave_osc.xmin` / `control/wave_osc.xmax` | `null` | `null`（Spine-only v1） | `[-1, 0, 1]` | normalized |

normalized expression sign 也属于 registry：`eye_open` 为 `0=closed,1=open`，`mouth_open` 为
`0=closed,1=max procedural open`，`mouth_form` 正值趋向 smile、负值趋向 frown，`brow_y` 正值向 canvas
上方抬眉、负值向下。TargetTransfer 必须实现这些语义并过 landmark/面积门，不能只让参数数值变化。

表中的两列分别冻结 `parameter_id` 与必需的 **export name**；持久化引用只使用 internal ID，并只能经
`GlobalExportSymbolTable` 的 Live2D parameter namespace 解析为
`ParamAngleX`。ControlSpec、binding plan 和 MOC3 不能一部分引用 internal ID、一部分绕过 symbol table
直接拼标准字符串。

`RigDocument.control_specs` 物化完整 v1 registry，而不是只写本 profile 当前存活的 control；clip/expression
只引用其中子集，Live2D liveness 再决定实际发射的 parameter。这样 profile/pruning 不会改变 control ID
universe，unused control 也不会被误写成 MOC3 dead parameter。

这里的 domain 是本产品的版本化模型契约，不是假装 Cubism 会替标准 ID 自动添加范围或行为。图像空间
`xmin/xmax` 不是 anatomical L/R，所以 v1 不得把眼/眉分侧 control 命名为 `ParamEyeL/ROpen` 或对应 brow
标准 ID；`.model3.json` 的 `EyeBlink` group 可以引用上述 custom IDs。未来只有在 C 持有显式、可验证的
`anatomical_side` observation/override 并升级 registry/schema 后才允许标准 L/R 映射，E 不能现场猜。
其他可选 brow/custom control 也必须先增加 registry row；不能由 E 看见一个 channel 后临时发明
min/default/max。
同一 Live2D `parameter_id` 在一个 Rig 中只能对应一个相同 domain 的 `ControlSpec`。

`ControlCurve` 只描述 control 随时间的值。`ControlBinding v1` 是 Rig 顶层、profile-independent 的模型
记录；每条至少包含
`binding_group_id, implementation_id, implementation_kind, implementation_rank, implementation_bundle_digest,`
`binding_id, control_id, target_id, property, transfer, visibility_branch_id, required_rig_facts, binding_digest`，其中
`TargetTransfer` 描述该 control 如何变成一个 canonical target property。一个 control 驱动多个 target
就有多条 binding；同一 implementation 内的 `(control_id,target_id,property)` 只能有一条。binding ID 从
`(implementation_id,control_id,target_id,property)` typed tuple 派生，
不能含 preset ID；clip/expression 的加入、删减或逐格式 omission 都不得复制或改写 binding。v1 transfer 允许：

- `affine_scalar/vector`：`output = output_at_default + gain × (control-default)`，用于
  rotation/translation/scale/opacity 等；
- `sampled_property` / `sampled_deform`：引用 C-owned、版本化且带 input/output digest（deform 另带
  topology digest）的 canonical piecewise-linear property/evaluator sample plan，用于 opacity、breath、
  blink/talk 和表情；不允许持久化一个没有版本或摘要的 Python callback。native implementation descriptor
  还把 variant opacity bindings 分成 visibility branches。所有 envelope sampled values 必须在 `[0,1]`；
  default 时每个 variant branch 都为 `0`，同一 `binding_group_id + control_id` 内互斥 variants 在每个 stop 的 envelope
  和必须 `≤1+1e-6`。普通 base drawable 始终为 setup opacity `1`，不属于这些 branch、也不能成为 native
  opacity target。不同 control 的 overlay 即使 coverage 重叠也不在这里错误相加，而是必须由 runtime
  compatibility matrix 标为 compatible/incompatible；

每个 transfer 在 `control=default` 时必须严格还原 Rig setup/rest property；affine transform 的
`output_at_default` 因而固定为 rotation/translation `0`、scale `1`，opacity 则必须等于目标自己的 setup
值（普通 drawable 为 `1`，native variant 为 `0`）；sampled deform/property 的 default sample 必须等于
rest vertices/property。违反时 C 以 `invalid_motion_clip` / `invalid_expression_preset`
失败，不能等 MOC3 default-rest gate 才发现共同语义本身就错了。

同一 `binding_group_id` 下的每个 `implementation_id` 是一个 atomic bundle：版本化 primitive registry
冻结其 expected binding-template ID 集合并形成 `implementation_bundle_digest`；所有
`required_rig_facts`/quality gates 均通过才是 eligible，缺一条 binding 就整体失败。C 按
`implementation_rank` 选择唯一 bundle，v1 的 registry 只允许普通单实现使用 `canonical=0`，替代实现使用
`native=10`、`procedural=100`；同 group 重复 rank、
未知 kind 或一个 format 同时选择两个 bundle 都是 `invalid_primitive_registry`/plan failure。选择结果和未选
原因写进逐 preset 与 set plan，D/E 不重新做“native 优先”的 if/else。比如 blink 的 native
closed-overlay opacity bundle 与 `eyewhite+irides+eyelash` procedural bundle 互斥，后者三层必须一起发射，不能混一层 native、
两层 procedural。

C 只把**完整且 item-eligible** 的 bundle 原子物化进 `RigDocument.control_bindings`；不合格 implementation
不留下半条占位 binding，其模板 ID 与失败原因进入 capability/format-plan diagnostics。完整性由
`implementation_bundle_digest` 对 registry expected set 复核，所以手工删掉一条落盘 binding 会使 Rig
validator 失败，而不是看起来像“本来就没有 native”。

一个 control 可以通过多条 `ControlBinding` 驱动多个 target channel，但一个 clip 对同一 control 恰好只有一条 control curve；
一个 Live2D parameter 在同一 motion 中也只能得到这一条曲线。D 通过 transfer 把 control curve 物化为
Spine target timeline，E 把 control curve 写一次到 `.motion3.json`，再把全部 target transfer 编译到该
parameter 的 deformer/ArtMesh/opacity bindings。这样 torso rotation 与 head translation 可以共享 idle
相位，却不会把 degree、pixel 和 parameter value 混成同一个数。无法因子化为明确 control + transfer 的
preset 必须在 C 以 `invalid_motion_clip` 失败。ExpressionPreset 同样只设置 absolute control value，再从
顶层 binding 解析目标；若 expression 内出现 transfer/channel 字段，schema 必须拒绝，不能把它当作覆盖。

`MotionClip v1` 的时间不是 binary float 秒数组，而是 `frame/sample_rate_hz` 的有理数。v1 固定
`sample_rate_hz=30`、所有 control key frame 为整数、`duration_frames>0`，且语义曲线是
piecewise-linear。版本化 preset 可在 30 Hz 网格上增加 control key 来逼近平滑解析曲线；每条 control
curve 的 frame 严格递增、值有限且位于 ControlSpec domain。loop clip 必须同时包含 frame 0 与 duration
frame，且每个 control 两端值严格相等。affine transfer 下 D/E 都不得重采样；`sampled_deform` 只允许使用
C preflight 已冻结的 approximation stops/times，并仍须对 canonical evaluator 过误差门，不能修改原 control
curve。未来 cubic/Bezier 需要升级 `motion_schema_version` 并新增两格式曲线 parity gate，不能在 v1 偷换
interpolation。

“固定 preset”还必须固定内容。`PresetLibrary motion-core-v1` 的 canonical descriptor 至少如下；tuple
均为 `(frame, control_value)`，同一行列出的双侧 control 使用相同 curve：

| preset | kind | frames / loop | control curves |
|---|---|---|---|
| `idle` | MotionClip | `120 / true` | `control/idle: (0,0),(30,1),(60,0),(90,-1),(120,0)` |
| `breath` | MotionClip | `90 / true` | `control/breath: (0,0),(45,1),(90,0)` |
| `head_nod` | MotionClip | `30 / false` | `control/head_nod: (0,0),(15,15),(30,0)` |
| `head_shake` | MotionClip | `45 / false` | `control/head_shake: (0,0),(10,-20),(25,20),(45,0)` |
| `body_sway` | MotionClip | `120 / true` | `control/body_sway: (0,0),(30,5),(60,0),(90,-5),(120,0)` |
| `blink` | MotionClip | `12 / false` | `control/eye_open.xmin` 与 `control/eye_open.xmax` 各为 `(0,1),(6,0),(12,1)` |
| `talk` | MotionClip | `30 / true` | `control/mouth_open: (0,0),(10,0.65),(20,0.2),(30,0)` |
| `wave.{side}` | MotionClip | `60 / false` | `wave_lift: (0,0),(15,1),(45,1),(60,0)`；`wave_osc: (0,0),(15,0),(25,1),(35,-1),(45,1),(60,0)` |
| `happy` | ExpressionPreset | n/a | `control/mouth_form=0.7, control/eye_open.xmin=0.8, control/eye_open.xmax=0.8, control/brow_y.xmin=0.15, control/brow_y.xmax=0.15` |
| `sad` | ExpressionPreset | n/a | `control/mouth_form=-0.7, control/brow_y.xmin=0.25, control/brow_y.xmax=0.25` |
| `surprised` | ExpressionPreset | n/a | `control/mouth_open=0.8, control/brow_y.xmin=0.8, control/brow_y.xmax=0.8` |

核心 geometry-normalized transfer 同样属于 descriptor，而不是实现默认值。`torso_width/height` 来自 A
冻结的 torso mask metric，`head_width/height` 来自 `head_core` mask metric；这些 metric/value 与来源摘要
写入 transfer，D/E 禁止重算 bbox：

- `idle`：`bone/torso.rotation = 1° × control`；孩子只通过骨骼层级继承，不额外复制同一 target；
- `breath`：以 torso/waist anchor 为原点，对 eligible torso deform region 使用
  `scale_x=1+0.01c, scale_y=1+0.015c` 的 canonical sampled deform；canvas `+y down` 下，位于 anchor
  上方的点随 `scale_y>1` 向上扩张。没有稳定 region/anchor 就没有 breath capability；
- `head_nod`：`head.translation_y=(c/30)×0.06×head_height`，
  `head.rotation=0.2c°`；`head_shake`：
  `head.translation_x=(c/30)×0.06×head_width`，`head.rotation=0.1c°`；
- `body_sway`：`torso.translation_x=(c/10)×0.02×torso_width`，
  `torso.rotation=0.2c°`；
- `wave.{side}`：令 visual side sign `s=+1` for `xmin`、`-1` for `xmax`，则
  `upper_arm.rotation=s×35°×lift`、`forearm.rotation=s×(20°×lift+15°×osc)`、
  `hand.rotation=s×5°×osc`；缺 hand bone 时整个 wave capability unavailable，不能删除 hand channel 后
  仍声称是同一 preset；
- blink/talk/expression 的 control 只选择本节前面定义的 layered/sampled deform target；若该 target
  quality gate 不通过则按 profile omitted/failed，不能换一套未版本化 bbox scale。

所有比例在 C 根据冻结的 Rig metric 物化成顶层 `ControlBinding.TargetTransfer` 并写 input/output digest；
同一 binding 可被多个 clip/expression 引用，但只物化一次。修改任一 key、
duration、loop、幅度、side sign 或 transfer 公式都必须升级 preset descriptor/version并失效 C 及下游，
不能只改 exporter。`blink/talk` 的 artifact kind 至此固定为 MotionClip；只有
`happy/sad/surprised` 是本版 ExpressionPreset。

`MotionRuntimeApplication v1` 还固定 parity/default 建议的播放环境：单 clip、weight/alpha=`1`、
fade-in=`0`、fade-out=`0`、无 crossfade、无其他 animation track 写同一 property。Live2D motion3 必须
显式写 `FadeInTime=0` / `FadeOutTime=0`，不能继承 SDK 或 Editor 默认值；Spine JSON 没有 mix-duration
资产字段，所以 `motion_manifest.json.runtime_application` 必须声明 `track=0,alpha=1,mix_duration=0`，
release harness 按此调用 AnimationState。下游当然可以自行 crossfade，但那是播放器合成策略，不能再用
该轨迹声称通过了 canonical single-clip parity。

表情组合的产品边界是“一个 base MotionClip + 至多一个 ExpressionPreset”。C 还要为每个 format 生成
确定性的 compatibility matrix；只有 matrix 为 compatible 的组合才适用下文双 track/runtime parity，其他
组合写入该格式的 `incompatible_with`，下游不得尝试后再猜结果。`dual_runtime_avatar_v1` 只要求每个
required expression 在 `idle` base 上通过，不要求 `talk+happy/sad` 并发；要求后者必须新增 profile。
单独存在的两个 artifact 不自动意味着可同时播放。
v1 matrix 的 native-overlay 判定还必须检查 control：同一 control 由 expression overwrite 时按下文
`suppresses_controls` 可组合；两个不同 control 的 selected overlay bundles 若 A 冻结的 component-level
coverage target/support 相交，且组合状态会让双方 variant opacity 同时大于 0，则固定 incompatible；不能
只因 xmin/xmax 共用一个未拆 Part ID 就误判，必须使用其已验证的 component partition。由此
`talk + happy/sad` 不会因为 target 是两个不同 variant ArtMesh 就被误报 compatible；模型可同时容纳资产，
不等于运行时可同时显示它们。

`RigTransformSemantics v1` 同时冻结 `TargetTransfer` 输出：translation 是 canvas pixel delta（`+x` 向右、`+y`
向下），rotation 单位 degree、**视觉顺时针为正**，uniform/non-uniform scale 是无量纲乘数。D 对 vector
delta 使用 `(dx,-dy)`、对 angle 使用 `-theta`；E 先用修正后的 canvas→root/local frame，再由 E0-core
签署 MOC3 `angle/base_angle` 到 canonical clockwise angle 的换算。两个 exporter report 都记录 transform
semantics version，runtime landmark 最终必须 inverse 回 canvas 与同一 canonical evaluator 比较；不能只
比较目标格式里的原始角度数值。

`ExpressionPreset` 是静态 control 目标状态，不伪装成某个格式的动画：它记录一个或多个
`{control_id, absolute_value}`、`application_mode` 和 capability 等级；value 必须落在对应
`ControlSpec` domain，target channel 只能由 `control_id` 反查同一组顶层 `ControlBinding`，不能另建
表情专用变形事实源或在 expression 内覆盖 transfer。v1 的
dual-runtime profile 只允许
`application_mode="overwrite_full_weight"`，并冻结运行时顺序为“先求值 base motion，再以 weight=1
应用 expression”。Spine 将其编译为独立同名 animation，公共 manifest 写
`runtime_application={track_role:"expression",track_index:1,loop:true,mix_blend:"replace",alpha:1,mix_duration:0,hold_until_cleared:true,apply_after:"base_motion_track_0"}`；
其中 exporter 要把 canonical absolute target 转成 Spine 相对 setup-pose timeline 数值，不能把“absolute”
误写成 Spine JSON 本身使用绝对坐标。Live2D 将同一 control absolute value 编译为已有参数 keyform 与
Overwrite `.exp3.json`，正式文件同样固定 `FadeInTime=0` / `FadeOutTime=0`，按已经冻结的
motion→save→expression→update 顺序应用。同一 control 同时存在于
base motion 与 expression 时属于**显式可组合**：full-weight expression 胜出，manifest 必须列出
`suppresses_controls`，两 runtime 都验证被压制 control 的最终 absolute value。两个不同 control 若直接
修改同一不可拆 non-rigid target，则仍是 `invalid_expression_preset` / `live2d_parameter_conflict`；C
preflight 必须失败或按 optional omission 冻结，不能依赖两个 runtime 各自的隐式优先级。

静态 expression 的 Spine 时间编码也不留给 exporter 猜。`SpineExpressionHold v1` 为每个实际 target
timeline 写两个数值完全相同的 key：`time=0` 与 `time=1/30`，因此 animation duration 固定为 `1/30 s`；
manifest/runtime adapter 在 track 1 以 `loop=true`、零 mix 持续应用，直到显式 clear。只写零时长单 key、
任意选择 1 秒 duration，或依赖某个 Spine Runtime 对 completed TrackEntry 的保留细节都非法。Live2D
`.exp3.json` 没有这条伪时间轴，expression manager 按前述零 fade 顺序保持 active；两边 clear 后的下一次
update 都必须恢复当时的 base-motion control 值。

Add/Multiply 仍由 E0-core 验证为 Live2D exp3 writer/runtime 的格式能力，但不属于 v1 双格式 expression
语义，也不能出现在 `dual_runtime_core_v1` / `dual_runtime_avatar_v1` 的 supported preset 中。未来若要
开放，必须为 Spine 定义可实载的等价 mixing contract 或显式做 per-format optional capability。若一个
表情只能由一段时间序列表达，它应是 `MotionClip`，不能硬塞进 expression。

“能力判断与格式编码分开”不等于把 omission 推迟给 exporter。C 在提交 Rig 前先对每个 exporter
运行版本化、无副作用的 `FormatModelPlan v1`，再对每个 `(preset_id, format)` 运行
`FormatCapabilityPreflight v1`：

- Spine model planner 检查骨骼拓扑、逐 component slot/setup attachment、weighted mesh、atlas
  reference 和全局 symbol namespace 的静态可编码性；
- Live2D model planner 检查 texture/page index、section/reference 容量、liveness 的静态上界，以及
  drawable count。v1 的 ArtMesh draw order 只使用 `0..1000`，所以 component count 必须 `≤1001`；
  映射固定为 `draw_order=component_draw_rank`。超限返回 `draw_order_capacity_exceeded`，不能 clamp、
  取模、复用相同值后依赖 Part/section 顺序，或等 E writer 才发现；
- preset planner 再检查所需 bone/slot/attachment/deform target；Live2D 侧使用同一 control/primitive
  selector 先按每个 binding group 的 `implementation_rank` 选择一个完整 atomic bundle，再用 liveness、
  non-rigid adaptive-stop/error/budget 算法检查 parameter/deformer/keyform 可实现性；不得跨 bundle 拼装。

逐项 supported 还不是最终承诺。C 随后对每个 format 运行 `FormatPresetSetPlan v1`，把所有候选的
`ControlBinding→PrimitiveCandidate` union 放进**同一个目标模型**检查 section、parameter binding、non-rigid
target 与全局预算冲突：

1. profile-required preset 全部先加入，按 canonical preset ID 排序只用于报告；任意 required-required
   冲突立即以 `missing_required_capability` 加具体格式原因失败，不能靠顺序删掉其中一个；
2. 其余 individually-supported optional preset 按冻结的优先级
   `body_sway < blink < talk < surprised < happy < sad < wave.xmin < wave.xmax` 依次尝试加入；这里 `<`
   只表示先选择，不是 draw order。加入后若破坏 model union，就把该 `(preset,format)` 冻结为 omitted，
   记录 `reason/conflicts_with/failed_primitive_keys`，已选集合不回溯；
3. v1 优先保留 `talk`。因此 Live2D 的 procedural `mouth_open` 与 `mouth_form` 若只能直接绑定同一个
   不可拆 ArtMesh，`happy/sad` 即使逐项可编译也会以 `live2d_parameter_conflict` omitted；Spine 若其模型
   结构允许二者作为独立 animation，则可同时保留。`surprised` 复用同一 `mouth_open` binding 时不构成
   第二个 non-rigid driver；
4. 严格 avatar profile 把 blink/talk/happy/sad/surprised 全部列为 required，故上述冲突会使 item
   fail fast。通过 coverage/role-scale/occlusion-intrusion 门的 native `occluding_overlay_v1` 会让 mouth-open/smile/frown 各自只驱动
   不同 variant drawable、普通 mouth base 不被任何 native control 写入，因此其 model union 合法；
   `talk+happy/sad` 仍在 runtime compatibility matrix 明确为 incompatible。没有这种 overlay 时，只有可靠
   target partition 或未来**另一个已验证 schema/profile**提供组合原语才通过；v1 不为追求通过率偷偷生成
   多参数 ArtMesh grid 或未 attested nested warp；
5. selection version、priority table、required/optional input set、完整 primitive-union digest 和最终顺序
   进入 `FormatPresetSetPlan` 摘要。每个 built-in optional preset 必须在表中恰出现一次；未知、重复或遗漏
   在 startup 以 `invalid_preset_registry` 失败。新增/重排 optional preset 必须升级 selection version并失效 C/D/E。

三类 preflight/selection 都不写 Spine/MOC3 文件、不调用商业 runtime，也不把格式 timeline 塞进公共 channel；
它们只返回 `{status, reason, planner_version, planner_input_sha256, planner_output_sha256, artifact_key}` 和
必要的 capacity summary。静态结果写入 `RigDocument.format_plans`，逐 preset 结果随 clip/expression 的
format decision 写入 Rig，set-plan descriptor/digest 也写入对应 format plan。profile-required format 的 model plan 为 unsupported 时 C 直接失败；required
preset 为 unsupported 时 C 以 `missing_required_capability` 失败，只有 optional preset 才冻结为 omitted。

D/E 调用同一 model/preset/set planner 纯函数复算并要求 input/output digest 与 Rig 完全相等，再进入 writer。planner
drift 使用 `format_plan_mismatch` 使 item 失败；结构编码、官方 runtime 或 validator 在 preflight 后失败也
是格式失败，不能把 C 已声明 supported 的 preset 临时改成 omitted。这样 capability 只有 C 一个 owner，
而 writer 仍只负责确定性物化与验证。

正式 dual profile 的 optional 策略冻结为 `optional_preset_parity="per_format"`，不是交集。C 阶段
根据版本化 exporter capability matrix 先把唯一决策写入 `RigDocument.clips/expressions`，再一次性投影
`motion_manifest.json`；D/E 只能执行 Rig 中属于自己的决策，不得临时增删 preset，也不得把 projection
当成可覆盖 Rig 的配置。每项至少包含：

```json
{
  "id": "clip/wave.xmin",
  "required": false,
  "supported_formats": ["spine_4_2"],
  "formats": {
    "spine_4_2": {
      "status": "supported",
      "planner_version": "spine-plan-v1",
      "planner_input_sha256": "sha256:...",
      "planner_output_sha256": "sha256:...",
      "preset_set_plan_sha256": "sha256:...",
      "incompatible_with": [],
      "artifact": "spine/skeleton.json#animations/wave_xmin"
    },
    "live2d_moc3_v4_00": {
      "status": "omitted",
      "reason": "live2d_joint_bend_requires_glue",
      "planner_version": "live2d-plan-v1",
      "planner_input_sha256": "sha256:...",
      "planner_output_sha256": "sha256:...",
      "preset_set_plan_sha256": "sha256:...",
      "incompatible_with": [],
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
| 常驻动作 | `idle`, `breath`, `body_sway` | resolved anatomical `torso`（`breath` 还需可用 torso deform region）+ 有效 mesh | synthetic `bone/root` 只保证格式拓扑，绝不单独满足动作 capability；幅度按 torso 宽高归一化，不写绝对像素 |
| 头部动作 | `head_nod`, `head_shake` | 独立 head/neck bone | 只写 bone rotate/translate timeline |
| 肢体动作 | `wave.xmin`, `wave.xmax` | 对应侧独立 upper-arm/forearm/hand 链与纹理 | Spine 4.2 正式/开发输出可生成；Live2D v1 以 `live2d_joint_bend_requires_glue` omitted，manifest 明确 `supported_formats=["spine_4_2"]` |
| 眼部动作 | `blink` MotionClip | 通过 occlusion coverage 门的闭眼 overlay variant，或可区分的 `eyewhite` + `irides` + `eyelash` mesh | 优先只驱动 closed-overlay opacity 的 native bundle；否则只允许下述 layered procedural blink，禁止压扁整只合成眼 |
| 口型动作 | `talk` MotionClip | 独立 mouth 层 + 合法局部 mesh | 周期性局部 deform；没有口腔新纹理时不得宣称音素级 lip-sync |
| 组合表情 | `happy`, `sad`, `surprised` | 各自 descriptor 实际引用的 brow/eye/mouth controls 均有合法 binding | 只组合非默认、可见的已有 overlay/deform channel；不得加入值等于 default 的占位 control 来制造虚假依赖，缺实际引用项则按 profile 失败或省略 |

单张 see-through 输出通常只有静态像素，**不包含闭眼、张口或笑脸的新纹理**。因此 expression capability 必须
区分 `native`（已有通过 coverage/role-scale/occlusion-intrusion 门的显式 occluding-overlay Part，只驱动 variant opacity）、`procedural`（只靠无三角翻转的 mesh deform）和
`unavailable`。默认不调用生成模型补脸，也不从别的角色借素材；否则批处理会稳定地产出身份
漂移。v1 的 native variant 只来自输入契约定义的 `rig_inputs/variants/manifest.json`；不接受 override
指向任意外部 PNG、绝对路径或 base64 旁路。variant set 使用独立摘要并使 A 及下游失效，但不改变用于
复用关节 overrides 的 base `target_input_fingerprint`。

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
- `dual_runtime_avatar_v1` 是 opt-in 严格 profile：在 core 基础上把 `blink`、`talk`、
  `happy`、`sad`、`surprised` 全部列为 required；v1 固定接受通过本文质量门的 `native` 或
  `procedural`，不存在调用方临时降低门槛的同名变体。缺少表情素材、layered deform 不合格，或
  required controls 在同一 Live2D 模型形成不可拆 non-rigid 冲突时 item 失败。native-only、要求
  talk+smile 同时叠加或启用多参数 mouth grid 都必须新增 profile/schema；
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

`SpineCoordinatePlan v1` 先冻结唯一根边界，所有 bone head/tail、mesh rest/bind-local vertex、diagnostic
landmark 与 animation translation 都必须走同一组纯函数：

```text
canvas_to_spine(x, y) = (x - canvas_width/2,
                         canvas_height/2 - y)
spine_to_canvas(x, y) = (x + canvas_width/2,
                         canvas_height/2 - y)
```

v1 固定 `1 Spine unit = 1 canvas pixel`，不读取 runtime loader 的外部 scale 猜资产单位；加载方可整体缩放
skeleton instance，但不能改变文件内契约。`bone/root` 在 Spine 中固定 `(x=0,y=0,rotation=0,scale=1)`。
plan version、canvas/input digest 和全部测试点 residual 写入 D report；forward→inverse 最大误差
`≤1e-6 px`，官方 runtime 渲染 landmark 回到 canvas 后容差 `≤0.1 px`。不能只翻 Y 而漏掉中心平移，
也不能对 bone 与 weighted bind vertices 使用不同原点。

`SpineBindPlan v1` 再冻结 hierarchy/local bind 数学。v1 所有 setup bones 使用
`transform="normal", scaleX=scaleY=1, shearX=shearY=0`。令 `h_b/t_b` 为经 `canvas_to_spine` 后的
world-space head/tail，`W_parent` 为从已序列化祖先 setup transform 正向重建的 world affine：

```text
local_head_b  = inverse(W_parent) * h_b
world_angle_b = degrees(atan2(t_b.y - h_b.y, t_b.x - h_b.x))
local_angle_b = normalize_to_[-180,180)(world_angle_b - world_angle_parent)
length_b      = norm(t_b - h_b)
W_b           = W_parent * T(local_head_b) * R(local_angle_b)
bind_vertex(b, p_world) = inverse(W_b) * canvas_to_spine(p_world)
```

root 的 `W_root=identity`，只有 root 可为零长度；其他 bone 的 length 必须大于几何 epsilon。weighted mesh
对每个 influence 分别写 `bind_vertex(b,p)`，不能只对 slot owner 求一次 local point；unweighted mesh 才对
slot bone 求一次。序列化后 parser 必须从 JSON 重建 `W_b`，证明每个 bone head、`W_b*(length,0)` tail
和所有 weighted setup vertices 回到原 canvas 的误差 `≤0.1px`。`SpineBindPlan` version/input/output digest
进入 D report/fingerprint；直接做 `child_head-parent_head` 或只减 parent angle 的实现不得通过 rotated-parent
fixture。

exporter 做且只做：

1. 从 `RigDocument.export_symbols` 读取 `GlobalExportSymbolTable` 的 namespaced Spine 子集；禁止在 D
   阶段构造、清洗、改变 namespace 或碰撞消解名称；
2. 通过 `SpineCoordinatePlan v1` 把 canvas center-origin、`+y down` 转为 Spine center-origin、`+y up`；
3. 每个 mesh component 生成一个独立 Spine slot 和一个 setup attachment，按 Rig canonical
   back-to-front `component_draw_rank` 排列；同 part 的 slots 必须连续、全部 setup attachment 均被选中并共享该 part
   atlas region；普通 part setup alpha=1，显式 native variant setup alpha=0。slot 的 setup bone 取该 component 所有 influence bones 的最近公共祖先（不存在时 root），
   只作 attachment owner；weighted vertex 中的真实 bone indices 不得被压成 slot bone。native variant
   Part 仍按自己的 component 生成相邻 sibling slot/setup attachment，setup slot alpha 固定 0，再由已选
   ControlBinding opacity bundle 做 variant-only、normal source-over 过渡，base slot 始终保持 setup alpha 1；这不是数学上的
   straight-color lerp，canonical renderer 与两 runtime 必须比较实际 composite。v1 不使用离散 attachment swap。禁止把同 part 多 component 塞成同一
   slot 的互斥 setup attachments，也禁止 D 按 attachment/tag/depth 重排；
4. `bone/root` 编码为 Spine identity root bone；其余 bone 严格按 `SpineBindPlan v1` 的 parent-world
   inverse/angle normalization 计算 local x/y、rotation、length；
5. 按同一 BindPlan 把 canvas-space rest vertex 分别转为**每个 influence bone**的 bind-local 坐标；
6. 按官方变长格式写 weighted vertices，扁平化 triangles；UV 必须通过版本化
   `Spine42UvAdapter` 从 C 的 `page_top_left_v_down` mapping 转换，并用官方 4.2 golden/runtime 固定
   atlas origin、V 轴与 region offset，不能直接复制 canonical `v_top`；
7. 读取 `RigDocument.texture_pages` 写合法 multi-page/multi-region atlas；以 byte copy 物化 page PNG，
   目标 `encoded_png_sha256`、region rect、padding/extrude 与 C 阶段计划完全一致，禁止 exporter 内
   decode/re-encode 或重新装箱；每个 atlas page 显式写 `pma:false` 并由 validator 与 Rig 的
   `alpha_mode="straight"` 交叉校验；attachment/region 合法同名时
   使用 Spine 默认 lookup，名称不同时显式写 `path`，validator 按 effective path 验证唯一 region；
8. 把 `MotionClip.ControlCurve + RigDocument.ControlBinding/TargetTransfer` / `ExpressionPreset` 映射为 Spine `animations` 下的
   bone/slot/attachment/deform timeline；MotionClip frame 必须按 `frame/30` 转秒。Spine 4.2 JSON 的线性
   segment 以**省略 `curve` 字段**编码，不存在合法的 `curve:"linear"`；写入 `"linear"`、`"stepped"`
   或 Bezier 数组都必须被 D validator 拒绝。affine transfer 禁止 exporter 重采样，sampled deform 只能
   使用 C plan 冻结的 approximation points，并把 canonical absolute sampled vertices 换算成 Spine
   setup-vertex offsets；canonical translation/rotation 分别按 `(dx,-dy)` / `-theta`
   转换。普通 motion 的 `track=0/alpha=1/mix_duration=0` 由公共 manifest 声明并由 harness 执行；
   expression animation 把 canonical absolute control target 换算为 Spine setup-relative timeline，并严格使用
   `SpineExpressionHold v1` 的 `0/1/30 s` 两个相同 key；应用顺序、expression
   track role、`loop=true`、`MixBlend.replace` 与 full alpha 由公共 manifest 的 runtime contract 声明，不能假装这些
   runtime 状态字段存在于 skeleton JSON；
9. 写 `skeleton.json`、`skeleton.atlas`、pages 和 `export_report.json`；公共
   `motion_manifest.json` 由 C 阶段写一次，不在 exporter 内复制事实源。

骨骼蒙皮、slot draw order、attachment path、
atlas region、required animation 缺任何一项都算导出失败。表情若声明 `procedural`，对应
deform timeline 必须存在并通过无 NaN、索引一致、关键帧拓扑不变和三角形不翻转校验。
默认有 error 级诊断时拒绝导出；`--allow-partial` 只允许已记录的刚性降级，不能绕过格式
validator，状态组合遵循上面的全局枚举。

验证分三层：纯 Python 结构 validator、与官方导出 golden fixture 对比、在有许可证的开发
环境中 opt-in 调用 Spine Editor 4.2/目标 runtime 实际加载并播放 required clips。普通 CI
不假装拥有商业软件。D 只验证并引用 C 阶段已经冻结的 TexturePagePlan、原样复制 canonical page，
不自行运行 packer 或 PNG encoder。

### Live2D Cubism runtime 导出契约

正式目标固定为 **SDK 可加载的 runtime bundle**，不是 Cubism Editor 工程：

- v1 artifact-layout schema 固定 basename 为 `model`，入口文件只能是
  `model.moc3`、`model.model3.json`、`model.cdi3.json`；item ID/display name 只进 manifest metadata，不能影响
  文件路径。未来开放自定义 basename 必须升级 layout schema 并重新做 path/symbol collision gate；
- `.moc3` 使用 header version `3`，即 **MOC3 V4.00**；
- `.model3.json` 使用 `Version: 3`，完整引用实际存在的 Moc、Textures、Motions、Expressions 和 Groups；
- 同包包含 `.cdi3.json`、PNG 纹理页、所有 manifest-supported `.motion3.json`，以及零个或多个
  manifest-supported `.exp3.json`；默认 core 若没有可用表情可以不生成 Expressions section，不能写
  空引用或虚构表情文件；
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
   引用全部使用全局映射，禁止 E 阶段重新清洗。按 Rig canonical `component_draw_rank` 映射为单调递增的
   Cubism draw order，禁止按 MOC3 section index/tag/depth 重排；同一 part 派生出的全部
   ArtMesh 共享该 part 的 atlas region，不重新裁纹理；
   普通 ArtMesh setup opacity=1，native variant ArtMesh setup opacity=0，并按 Rig draw bundle 的连续 rank
   紧随 `draw_anchor_part_id`；只能由已选 native opacity binding bundle 改变，不能靠 motion 文件临时创建/隐藏 mesh；
2. 从 canvas-space Rig 构建版本化 `Live2DCoordinatePlan`。PPU 归一化只用于 canvas→root model
   边界；每个 WarpDeformer、RotationDeformer 和 ArtMesh 都按其**直接父节点**转换到对应 local
   frame，禁止把同一 root 公式套到整棵树。逐层 forward/inverse、父 ID、frame kind、输入摘要和
   round-trip residual 写进报告；
3. 从 `RigDocument.texture_pages` 确定性分配 texture index，并由 attested `CubismV400UvAdapter` 把
   canonical `page_top_left_v_down` mapping 转成 MOC3/Core UV，再把 canonical PNG 原样复制到 Live2D
   texture path；页数必须 `≤4`、每页必须 `2048×2048`、目标 SHA 必须等于
   `encoded_png_sha256`，且所有 texture reference 能从 `.model3.json` 闭环解析；
4. parameter identity/domain 只从 `RigDocument.control_specs` 读取；其 `ControlRegistry` 已在 **ID 命名层**
   为 `control/head_shake→ParamAngleX`、`control/head_nod→ParamAngleY`、
   `control/body_sway→ParamBodyAngleX`、`control/idle→ParamAutoIdle` 等 control 冻结 standard/custom ID、
   min/default/max 与 unit。E 不得从 target property 或 preset 名重推 parameter。适用的
   `ParamAngleZ`、`ParamBreath`、`ParamMouthOpenY`、`ParamMouthForm` 和 brow parameter 也必须先有 registry
   row；眼/眉的 `xmin/xmax` 在 v1 固定使用 custom parameter，不能冒充 anatomical L/R。无法对应标准含义时使用 C 已冻结的稳定 ASCII custom parameter，
   并在 `.cdi3.json` / report 中说明。标准 ID 只改善
   EyeBlink/LipSync group、面捕和第三方 runtime 的语义兼容，**没有任何内建变形行为**；core v1
   的 `head_nod/head_shake` 仍只是固定的 2D pivot/translate 风格化动作，不得因使用
   `ParamAngleY/X` 就宣称重建了新视角、3D yaw/pitch 或可泛化的面捕行为；
5. 先从 `RigDocument.control_specs/control_bindings/clips/expressions` 中经 projection validator 确认的 Live2D-supported
   control curves、absolute control targets 与唯一 target transfers 解析确定性的 `Live2DBindingPlan`，再由该计划
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
   由 E0-core 相对 `RigTransformSemantics v1` 的 clockwise-positive angle 固定；不能在 exporter 内交换
   旋转/平移顺序或把目标格式原始 angle 数值直接当 Rig angle；
6. 把 `breath` 等非均匀结构形变编译为单参数 WarpDeformer，把对应 RotationDeformer 链作为其
   子级；正式 v1 只允许 E0 已签署的 axis-aligned rectangular grid，并固定
   `quad_transforms=true`。任意曲面 grid、`quad_transforms=false`、warp-under-warp 或非均匀
   scale/shear 都不得伪装成已支持的 RotationDeformer/structural warp，必须以 capability 不支持拒绝；
7. 正式 v1 只对 breath/其他结构 warp、blink/talk、表情和其他真正改变局部形状的已支持 channel
   烘焙 ArtMesh/warp keyforms 或 opacity binding；LBS 混合带只能由 E0-S 实验 compiler 产生。第 4 条
   决定参数名，第 5-7 条决定原语和 binding；
   `.motion3.json` 只负责驱动已经绑定的参数，不能替代任何 binding；
8. 把 `blink`、`talk` 和其他时间序列写为 `.motion3.json`；把 `happy/sad/surprised` 写为
   `.exp3.json`。每个 clip/control/parameter 恰好写一条 motion3 curve，segment 必须逐项对应 Rig 的
   `ControlCurve` `frame/30` linear keys；多个 target transfer 只能共享该 curve，不能重复写互相覆盖的
   parameter segment，也禁止 E 自行拟合 Bezier。每个正式 motion3 显式写 `FadeInTime=0`、
   `FadeOutTime=0`，per-curve fade 也不得重新引入非零权重包络；
   需要网格变化的 expression 同样先生成 parameter keyforms；正式 dual-runtime v1 的
   `.exp3.json` 只写 full-weight Overwrite 目标。Add/Multiply 只存在于 E0 conformance fixture 或未来
   显式的 per-format profile，不能混进 dual-runtime supported preset。Cubism runtime JSON 不使用
   JCS/minified bytes；统一经 `cubism_runtime_json_bytes()` 写 ASCII、`indent=2`、stable key order 与
   terminal newline，确保每个数字在 `]`/`}` 前先遇到官方 Framework 5-r.5 可识别的换行终止符；
9. 在 `.model3.json` 注册 motions、expressions，以及实际存在的 `EyeBlink` / `LipSync` 参数组；
10. 写 `.moc3`、JSON、纹理、参数映射和 `export_report.json`；report/公共 manifest 必须携带
    `texture_runtime_contract` 与已验证 target runtime loader mode，不得声明未生成的 physics/pose 文件，
    也不得向标准 model3 私加一个 runtime 不认识的 PMA 字段。

#### Live2D deformer 活性剪枝

先把 `RigDocument.clips/expressions` 中
`formats.live2d_moc3_v4_00.status="supported"` 的每个 control curve/expression control target 与 target
channel 通过 `control_id` 反查 `RigDocument.control_bindings` 后解析成 `Live2DBindingPlan`，并断言公共
`motion_manifest.json` projection 完全一致。每个 supported control 的每条 applicable binding 必须恰好对应
一个计划项，每个 supported motion control 必须恰好对应一个 parameter-curve record；
计划项至少记录
`preset_id/control_id/control_curve_digest/binding_group_id/implementation_id/binding_id/rig_target_id/transfer_digest`
和非空
`bindings=[{candidate_id, parameter_id, primitive_kind, primitive_target_id, stack_rank}]`。每个 binding
必须是 `RigDocument.primitive_candidates` 中一个 immutable record 的实例化结果。RotationDeformer 的
`primitive_target_id` 必须引用 typed `(bone_id, parameter_id)` instance key，不能只写 bone ID；
`stack_rank` 对非 rotation primitive 为 `null`。一个 control/channel 可以合法驱动多个 ArtMesh/opacity target，
但不能有两份互相矛盾的计划项；无 binding 时先报
`live2d_parameter_binding_missing`。同一 motion 中若两个 control curve 解析到同一 `parameter_id`，或同一
control 被物化为两条不同 parameter curve，必须在 C 以 `invalid_motion_clip` 失败。计划只由 Rig、preset
和版本化原语选择规则推导，**不得读取 writer
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

| control ID | parameter ID | 同 bone `stack_rank` |
|---|---|---:|
| `control/body_sway` | `ParamBodyAngleX` | 100 |
| `control/idle` | `ParamAutoIdle` | 200 |
| `control/head_shake` | `ParamAngleX` | 300 |
| `control/head_nod` | `ParamAngleY` | 400 |

表中的 parameter ID 只是便于审查的 resolved export name；序列化的 `RigidDriverRegistry` row 只保存
`control_id`、rigid primitive kind 与 stack rank，再经对应 ControlSpec 取得 parameter internal ID/name。
它不拥有第二份 control→parameter mapping。

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
                        (canvas_height/2 - y) / PPU)
```

这里的 Y 翻转已经由 Core drawable 顶点与 canvas evaluator 双向实测：Cubism Core 的 root/model 坐标
`+y up`，LayerDiff canvas 为 `+y down`，`root_to_canvas` 必须使用精确逆式。Revision 22 还完成了
warp/rotation 的限定 E0；`coordinate_schema_version="live2d-frames-v1"` 现在可以进入
`export_report.json` 和 export fingerprint，但只覆盖下表明确写出的 v1 域，不得向 MOC3 私加 metadata
section，也不得把未测的一般 warp 行为塞进同一版本：

| 节点/数据 | 编码 frame | 规则 |
|---|---|---|
| root ArtMesh/keyform | `ROOT_MODEL` | 使用上面的 PPU-normalized root model 坐标 |
| WarpDeformer grid keyform | v1 为 `ROOT_MODEL` | 正式 v1 只发射 root 下的 axis-aligned rectangular structural warp，固定 `quad_transforms=true`；三个 breath key 的四角都显式写入 root frame |
| WarpDeformer child input | `WARP_LOCAL` | 该限定矩形的输入域为 `[0,1]×[0,1]`。`quad_transforms=false` 的新版非简单插值不是普通 bilinear，v1 禁止使用；一般曲面 warp inverse 未签署 |
| 首个 RotationDeformer origin | 直接父 warp 的 `WARP_LOCAL` | origin 随 warp grid 移动成为 canvas pivot；禁止直接写 canvas/ROOT_MODEL 值 |
| 嵌套 RotationDeformer origin | 父 rotation 的 `ROTATION_LOCAL` | 使用父 rotation authoring-local 单位，按明确 parent index 嵌套；不能反向从数值范围猜单位 |
| RotationDeformer child input | `ROTATION_LOCAL` | 固定 `T(origin)·R(angle)·S(scale)`。实测中 warp 改变 pivot，但不会把 warp 的非均匀伸缩 Jacobian继续乘到 rigid local vector；转换尺度由 rotation keyform 的 `scale` 显式承担 |
| ArtMesh keyform | 直接父 deformer 的 input/local frame | 无父节点才允许 `ROOT_MODEL`；挂在 warp 下使用 `0..1`，挂在 rotation 下使用相同 authoring-local 单位，不得再次套 root PPU 公式 |

warp-local 线索最初来自 StretchyStudio 对 CMO3 Editor/Hiyori 的逆向记录，Revision 22 的签署依据则是
本项目 writer 生成的 `root→warp→rotation→rotation→ArtMesh` 在官方 Core 06.00.0001 中完成 rest、
端点、九点 angle+origin 与多参数矩阵的最终 vertices parity。这个证据仍不足以证明任意 warp grid、
`quad_transforms=false`、warp-under-warp 或 rotation-under-curved-warp；新增这些能力必须升 frame schema、
扩 E0 并重签，不能用当前 `≤0.1 px` 结果外推。

E0-core 证明的是**窄的 frame/runtime 语义契约**，不是“这份 compiler 源码每一行都被认证”。
`live2d_frame_contract_digest` 对 canonical descriptor 求 SHA-256，descriptor 只包含：

1. `coordinate_schema_version`、frame table 和 canvas/root/warp/rotation/ArtMesh frame kinds；
2. 只承载 `encode_point/decode_point`、`T·R·S`、`rotation-stack-v1` comparator/parent-chaining、
   `CubismV400UvAdapter` 与相关 binary field codec 的 side-effect-free semantic kernel 模块源码摘要、
   版本、语义 descriptor 与 synthetic 纯函数测试向量；production `RigidDriverRegistry` rows 明确排除；
3. 影响 deformer parent、keyform binding、origin/positions 的 MOC3 header/SOT section-layout descriptor
   版本与摘要；
4. E0 fixture/golden SHA-256、default-rest/round-trip/runtime parity，以及非对称纹理 UV/最终采样 parity
   结果；
5. 逐个通过同一 E0 矩阵的 approved Core binary SHA-256 allowlist，以及执行 Core call/assertion 顺序的
   `e0_validator_protocol_digest`；不把 harness 的日志/报告实现纳入 gate。

descriptor 的序列化本身不是 E0 再决定的开放项。`Live2DFrameAttestation v1` 固定为一个 RFC 8785 JCS
JSON object：顶层只有 `schema_version`、`contract_descriptor`、`live2d_frame_contract_digest` 和
`provenance`。digest 精确等于
`SHA256(JCS(contract_descriptor))`；digest 字段自身与整个 `provenance` object 不进入被摘要子树，禁止把
完整文件自哈希。`contract_descriptor` 固定包含：

```text
coordinate_schema_version
frame_kinds[]
semantic_kernels[]
layout_descriptors[]
pure_vectors[]
e0_fixtures[]
invariants[]
approved_core_binaries[]
e0_validator_protocol_digest
```

数组顺序分别固定为：frame kind 的显式 `ordinal`，其余依次按 `kernel_id`、
`(section_index,descriptor_id)`、`vector_id`、`fixture_id`、`invariant_id`、
`(platform,arch,core_sha256)` 升序；每项 ID 必须唯一，validator 先拒绝乱序/重复再求 digest。
`semantic_kernels[].source_sha256` 使用 `KernelSourceDigest v1`：源码必须可解码为 UTF-8，去 BOM，把
CRLF/CR 仅规范为 LF 后按 UTF-8 重编码并取 SHA-256；不删除注释、空白或尾随换行，避免“规范化”藏掉实际
kernel 改动。layout/vector/fixture/invariant payload 全部是 JCS 子对象，不接受 JSON 字符串内再塞一份未解析
JSON。Core allowlist 只记录平台、架构、公开版本标识与 binary SHA，不记录机器绝对路径。数值继续遵守
本文 I-JSON/finite/negative-zero 规则。

semantic kernel 必须是隔离的小模块，禁止日志、报告渲染、CLI、错误文案和业务 orchestration；这些
非语义代码留在外层 compiler/writer。这样 kernel 的任意实现变化都会失效 attestation，而外层无关
改动不会。不能为了少跑 E0 把实际坐标、UV 或 layout 逻辑移出 kernel。

Revision 22 已通过 E0-core，并生成随 compiler package 固定的
`module/auto_rig/export/live2d/attestations/live2d-frames-v1.json`。首个 allowlist 只含官方 SDK 5-r.5
Core `06.00.0001` 的精确 Windows x64 SHA-256；attestation 包含上述 gate descriptor/digest，
另有不参与 coordinate gate 的 `provenance`，记录完整 compiler/writer version、源码摘要、构建
commit、完整 SDK harness 源码摘要和可选 `SOURCE_DATE_EPOCH`；禁止写 wall-clock 生成时间导致相同签署内容
每次产生不同 package bytes。完整源码摘要仍进入 export cache fingerprint 和报告，
但改日志、增加 report 字段或修复与 frame 无关的错误路径，不会仅因 provenance 变化强迫重跑
E0。反过来，改 frame codec、transform/
stack order 或相关 layout descriptor 必须改变 contract digest；普通 CI 还要运行 attestation 内的纯函数
向量与 parser golden，防止实现变了却漏升 descriptor 版本。

这里的“签署”是可复现的摘要绑定，不额外引入 PKI，也不是每个 item 的产物。扩大 Core 兼容性不能
只写 semver range：每个新增 Core binary 都必须按相同 validator protocol 实际通过 E0，加入
allowlist 后生成新 attestation。

attestation 是技术兼容证明，不分发也不授权使用 Core/SDK。generator 要求显式确认 SDK/Core 不进入
wheel、仓库或模型交付，并把“组织是否需要/已经取得 Release License”记录为外部发布门；该记录不构成
法律意见，也不能由 technical digest 替代。

启动 gate 按以下顺序执行，且都先于输入 item 枚举：

1. 所有 auto-rig job 已在公共 startup gate 验证 packaged draw-order DAG、`ControlRegistry`、
   preset/profile/diagnostic/primitive registry 与 selection priority；structural/release job 再加载 parameter/
   `RigidDriverRegistry`，并证明每个 rigid row
   引用的 control 存在且唯一解析到一个 Live2D parameter。built-in rank 重复或 rigid reference 冲突返回 job-level
   `invalid_rigid_driver_registry`，其余 registry 按自身 `invalid_*_registry` 返回，均不创建 item
   staging/report；
2. 所有 structural/release job 重算本地
   `live2d_frame_contract_digest` 并执行坐标/UV 纯函数向量；attestation 缺失、descriptor/digest/vector
   不匹配时
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

纯 opacity sampled binding 不套像素宽度公式；它在同样的区间采样点要求 canonical 与 runtime scalar
绝对误差 `≤1/255`，并继续受 `[0,1]`、variant-envelope `sum≤1`/default=0、coverage/role-scale/occlusion-intrusion 和 17-stop 门约束。若 canonical transfer
本身是 piecewise-linear，compiler 必须保留所有折点而不是用误差容差删掉语义状态。

v1 的 `max_stops_per_binding=17` 作用于每个唯一 sampled ArtMesh/warp/opacity
`(parameter_id, primitive_target_id)` binding，端点和必要的 default stop 都计入 17；它不是全模型共享
17 个点，也不是每个 preset 可为同一 binding 再分配 17 个点。同一 immutable keyform payload 被多个
preset 引用时只能按完全相同的 payload digest 共享一次。
`total_baked_vertex_positions` 精确定义为
`Σ(ArtMesh binding stop_count × emitted_vertex_count) + Σ(Warp binding stop_count × emitted_control_point_count)`；
它的上限为 `1,000,000`，只统计 ArtMesh/warp 的 non-rigid keyform positions，不统计 RotationDeformer 的
七个标量 keyform 字段；`.moc3 ≤ 64 MiB` 仍约束
整个文件。C 的 Live2D preflight 中，任一 optional non-rigid preset 在上限内达不到误差阈值就冻结为
omitted；required preset 则以 `live2d_interpolation_error` 或 `live2d_keyform_budget_exceeded` 使 C 失败，
不能偷偷降低 stop 数。E 复算得到不同结论只能报 `format_plan_mismatch`，不能在 writer 阶段改 status。
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

job bootstrap/startup 与 unsafe item discovery 使用独立的 batch/job-level result，不写进任何 item 的
`rig/report.json`：

```text
invalid_draw_order_registry
invalid_control_registry
invalid_preset_registry
invalid_primitive_registry
invalid_profile_registry
invalid_diagnostic_registry
invalid_rigid_driver_registry
live2d_coordinate_schema_unverified
live2d_release_gate_unavailable
live2d_core_unattested
item_discovery_failed
```

除 `item_discovery_failed` 在安全枚举某个候选 root 时产生外，其余错误都在枚举 item 前返回非零；它们都
没有 item/part/joint ID，`continue_on_error` 的已枚举-item 语义也不适用。
这组 job-startup code 是 runner 内的最小 bootstrap enum，不依赖待校验的 `DiagnosticRegistry`；否则 registry
损坏时连 `invalid_diagnostic_registry` 自身都无法编码。bootstrap result 只含 code、registry kind/version 和
稳定 repair ID，不复用 item diagnostic schema。
所有 profile 都先验证 packaged/versioned `DrawOrderPolicy` DAG、`ControlRegistry`、preset/profile/diagnostic
registry 与 primitive registry 的 ID/domain/reference/priority；Live2D structural/release profile 再验证
`RigidDriverRegistry` 和这些表的引用一致性。错误按对应 `invalid_*_registry` code 返回。v1 不接受临时用户扩展；
若未来开放扩展，它仍必须先合并并通过同一 startup validator，不能把配置错误下放成逐 item 失败。

`terminal_finalization_failed` 是唯一允许只出现在 batch item result、而不保证能进入公共
`rig/error.json` 的枚举后错误：它表示 G 自己无法可靠写入 terminal artifact/manifest。runner 必须保留
item ID、原始 stage failure 摘要和系统异常供批次汇总，但不得伪造一个已经持久化的 error record；没有
有效 G manifest 就永远不是 completed，下次运行必须重试 finalization 或最早失效 stage。

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
internal_id_collision
export_name_collision
missing_export_symbol
missing_required_capability
invalid_motion_clip
invalid_expression_preset
format_plan_mismatch
draw_order_capacity_exceeded
texture_budget_exceeded
native_variant_ineligible
native_variant_texture_budget
native_variant_drawable_budget
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

在 C 成功前，同一诊断 record schema 写入 owner stage 的私有 report/`failure.json`；失败绝不写 success
manifest。A/B 失败不会为了产出
错误信息而创建一个不完整的公共 `rig/report.json`，而是由 G 把规范化 failure record 发布到公共
`rig/error.json`。C 成功时才把 A-C diagnostics 快照到其 owned report，之后 D/E 仍只写各自 export
report；任一 A-E item failure 最终也由 G 更新同一个公开错误契约。F 实验失败只进 F report/job result。

每项带 `code/severity/stage/item_id/repair={template_id,args}`；`args` 只能是 registry 声明的 typed stable
fields，面向 GUI 的本地化句子在读取时渲染、不进入 record/digest。只有错误确实归属于某个实体时才带 typed
`entity_ref={kind,id}`，不能为 item-global export/cache 错误伪造 part/joint ID。正常模式出现 error 返回非零；
批处理的 `continue_on_error` 只隔离 item，不能把失败 item 记成 completed。
`DiagnosticRegistry v1` 为每个稳定 code 冻结允许的 stage、severity、terminal effect、`retry_policy`、必需
context keys 与 repair template ID；report/failure/G 只能物化 registry row，不能在 catch block 临时改 severity
或 retry 语义。重复/未知 code、同一 code 两套 policy 或 repair template 缺失在 startup 以
`invalid_diagnostic_registry` 失败。
`live2d_joint_bend_requires_glue` 在正式 v1 固定为 `warning + preset_omitted`；
`native_variant_texture_budget` 固定为 A 阶段 `warning + native_bundle_rejected`，只表示 mandatory base plan
已通过但该 atomic optional overlay group 无法在四页内加入；C 随后按同一 rejected fact 选择 procedural/omit，
strict 无替代时才另报 `missing_required_capability`。它不能代替 base-set 的 error 级
`texture_budget_exceeded`。
`native_variant_drawable_budget` 使用同一 severity/effect，只表示加入 group 会让 projected component/drawable
count 超过 1001；它不能靠共享 draw-order 数值绕过，base set 自身超限仍由 C 的
`draw_order_capacity_exceeded` hard gate 确认。
`native_variant_ineligible` 同样是 A warning/capability rejection，必带冻结 reason enum
`coverage_leak | alpha_mass_ratio_exceeded | occlusion_intrusion | component_partition | bundle_incomplete` 以及对应
metrics/role/bundle ID；envelope 相关 metrics 至少含 base/variant alpha 整数和、共同 scale kind、仅供诊断的
base support bbox/aspect ratio、最终 radius、完整 `intrusion_by_part`、face/other aggregates 与 intrusion ratio，
不能只留一个 pass/fail；
`spill_mass_diagnostic` 只进入 metrics，不是 rejection code；它不表示
manifest 语法非法。core 可选 fallback，strict 无替代时由 C 另报 `missing_required_capability`。
`internal_id_collision` 只用于已枚举 item 内两个不同 canonical identity records 产生同一个完整 internal ID，
固定为 error 且禁止数字后缀/遍历顺序兜底；packaged registry 自身的重复 semantic ID 仍是对应 startup
`invalid_*_registry`，不能下放成首个 item 的 collision。
`live2d_skinning_seam` 只允许出现在 E0-S，固定为 experimental warning。当前没有 profile 把
独立肘/膝弯曲列为 required，因此两者都不得升级为 item error；未来 profile 若要求该 capability，
应在 profile 选择时以 `missing_required_capability` fail fast，直到 Glue E0 gate 已被版本化启用。
`live2d_parameter_conflict` 若由 C 的 `FormatPresetSetPlan` 命中 optional candidate，固定为
`warning + preset_omitted` 并列出 `conflicts_with`；同一冲突涉及 required preset 时同时产生 error 级
`missing_required_capability` 并使 C 失败。E writer 才首次发现的冲突永远是 `format_plan_mismatch`/格式失败，
不能在 E 降级成 warning。
`live2d_coordinate_schema_unverified` 是 job-startup error，只表示 attestation 未生成、
frame contract descriptor/digest 或纯函数向量不匹配；它不比较完整 compiler/writer provenance。
release tier 缺 Core/SDK 用 `live2d_release_gate_unavailable`，存在但不在 E0 allowlist 用
`live2d_core_unattested`。draw DAG、control domain/ID 和 rigid rank/reference 的 startup 错误分别只用
`invalid_draw_order_registry`、`invalid_control_registry`、`invalid_preset_registry`、
`invalid_primitive_registry`、`invalid_profile_registry`、`invalid_diagnostic_registry`、
`invalid_rigid_driver_registry`。
`live2d_coordinate_roundtrip_failed`、`live2d_default_pose_mismatch` 和
`live2d_deformer_order_conflict` 才是具体 fixture/item 的编译、binding 或实载错误。

### 集成点与实施顺序

- CLI：`module/auto_rig/cli.py`，复用项目批处理、Rich progress 和 continue-on-error UX；但
  `--skip_completed` 必须调用 auto-rig `StageGraphValidator`，不得复用 see-through
  `detect_resume_stage()`/“最终文件存在即完成”的 predicate，且只允许正式 release profile 使用；
  dev/structural 只提供 stage resume。每个已枚举 item 的 A-E production-stage 失败通过 G 写
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
packaging、模型 inventory 和测试注册。实现计划必须按 E0、A/B、C、D、E、G 与独立实验 F 分提交，
不能用一个巨型 PR 同时落地几何、preset、Spine、Live2D、终态协议和姿态模型；第一份实现提交应是
隔离的 E0-core fixture/writer
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
| A 使用 packer、却只有 C 跟随 packer fingerprint | packer 升级后 A dry-run 陈旧，C 才报 plan mismatch | texture-plan/MaxRects 契约同时进入 A/C fingerprint；PNG encoder 仍只进入 C |
| B/C 把同一 `rig.json` 都声明为 output | C 改写后 B 摘要永久失配，昂贵 mesh/weights 每次重算 | stage-owned path 集合两两不交；B 写私有 `RigGeometryCache`，C 独占并一次性写完整 RigDocument |
| `export_manifest.json` 存在即触发 item skip | D/E encoder、validator 或 artifact 已变化仍交付旧 completed 包 | G 成为真实 terminal stage；`skip_completed` 递归验证 A-E/G fingerprints、manifests 和 artifact SHA，陈旧 G 先撤销终态 |
| dev/structural 成功被强迫写 G terminal | 非发布报告被误标 completed，或无错误却被写成 failed | G success 只用于正式 release；非发布成功停在活动 stage report，A-E failure 仍由 G 公共发布，F 实验失败除外；`skip_completed` 对非发布 job 禁用 |
| A/B 失败只留在可删 cache/批次日志 | 无 per-item 可寻址诊断，`diagnostic → overrides` 闭环断裂 | G 独占公共 `rig/error.json`；任一已枚举 item 的 A-E production-stage 失败都发布稳定 failure record，成功时与 export manifest 互斥；F 实验失败不改产品终态 |
| 旧 page/motion/expression 留在可变输出目录 | profile 或页数收缩后仍交付未声明 artifact | stage manifest 记录精确 inventory；旧 commit marker 先失效，清除 owner namespace 中的 obsolete 文件后再提交新 manifest |
| `motion_manifest.json` 与 Rig 分别决定 capability | 两份文件可各自合法却让 D/E 导出不同 preset | manifest 只作绑定 Rig SHA/语义 digest 的确定性 projection；D/E 只从 Rig 选 binding 并交叉验证 projection |
| exporter 在 C 已承诺 supported 后临时 omit | Rig、manifest 与实际 artifact 分叉，optional 行为不可复现 | C 运行共享 FormatCapabilityPreflight；D/E 复算 plan digest，writer/validator 失败只能使 item 失败 |
| 把 Live2D exp3 blend mode 当成跨格式 expression 语义 | Spine 与 Live2D 在 base motion 上组合出不同表情 | dual-runtime v1 只允许 full-weight overwrite；Spine application contract 写进 manifest，Add/Multiply 仅作 Live2D E0 conformance |
| MotionClip 未冻结 key 间插值 | ID/duration/loop 一致但两格式轨迹不同 | v1 使用 30 Hz integer-frame keys + 显式 linear segment；两 runtime 按 frame/中点与 canonical evaluator 比对 |
| depth 没有共享 draw-rank policy | 两 exporter 对眼部/头发/交叉肢体各自排序 | A 冻结 256-bucket part rank，B 连续展开 component rank；D/E 只映射方向，默认 composite 实载对比 |
| 多 component 共用一个 Spine slot | setup pose 只显示一个互斥 attachment，角色缺块 | 每 component 一个连续 slot/setup attachment，共享 part region；slot bone 取 influences 最近公共祖先，权重不丢 |
| anatomy root 依赖 pelvis/spine | 半身/头像没有共同根，parent fallback 与 slot owner 无效 | 永久 synthetic identity `bone/root` 与可选 `lower_torso` 分离；只有 root 允许空 joints/零长度 |
| Spine 只翻 Y、未冻结原点/单位 | skeleton 整体偏移，bone 与 weighted mesh 使用不同 bind frame | `SpineCoordinatePlan v1` 固定 center origin、1 unit/px 与双向公式；root=(0,0)，parser/runtime landmark round-trip |
| Live2D root 沿用 canvas `+y down` 或 rotation 正号未定义 | 两格式对同一动作上下/旋转方向相反 | root 使用 `(H/2-y)/PPU`；canonical rotation 为视觉顺时针正，Spine 写负号，Live2D 由 E0 签署后 inverse 回 canvas |
| 把 Spine linear 写成 `curve:"linear"` | JSON 字段结构看似合理但不符合 4.2 timeline 契约 | MotionClip 显式声明 linear；Spine JSON 通过省略 `curve` 编码，其他值由结构/runtime mutation tests 拒绝 |
| component rank 超过 Cubism draw-order 域 | 重复/clamp 后前后层不确定 | C 的 Live2D FormatModelPlan 要求 drawable `≤1001`，连续 rank 原值映射到 `0..1000`，超限 fail fast |
| semantic 遮挡只作 depth tie-break | 异常 depth 让刘海、眼部或耳饰前后颠倒 | behind→front DAG 作为硬约束；Kahn ready set 才用 depth/stable ID，registry startup 验证无环 |
| control parameter 与 target property 共用一组 keys，或 transfer 被复制进各 preset | 多 target motion 出现 degree/pixel 冲突；expression 没有可复用映射；同一 Live2D parameter 多条曲线 | `ControlSpec/ControlCurve/ControlBinding/TargetTransfer` 分层；binding 顶层单写，每 control 一条 curve，motion/expression 只引用 |
| `xmin/xmax` 被命名成 Live2D anatomical L/R parameter | 镜像/侧背角色眼眉控制接反且第三方面捕语义虚假 | v1 使用 custom XMin/XMax IDs，EyeBlink group 引用 custom IDs；标准 L/R 等待显式 anatomical-side schema |
| 固定 preset 只冻结名称、不冻结内容 | 不同实现的时长、幅度和动作轨迹都不同却使用同一版本 | `motion-core-v1` 固定 control keys、loop/duration 与 geometry-normalized transfer；任何内容变化必须升版本 |
| blink/talk 在 clip 与 expression 间分类漂移 | manifest 路径、runtime API 和双格式 parity 无法闭环 | blink/talk 固定为 MotionClip，happy/sad/surprised 固定为 ExpressionPreset，projector 验证 artifact namespace |
| motion fade/mix 使用 runtime 默认 | canonical keys 相同但首尾权重与轨迹不同 | motion3 显式零 fade；Spine manifest/harness 固定 alpha1、mix duration0，parity 只声明 single-clip 环境 |
| Spine bind-local 只做坐标/角度相减 | rotated parent 下 bone tail 与 weighted vertices 脱离 setup pose | `SpineBindPlan` 使用 parent/bone world affine inverse，逐 influence 生成 local point并正向重建验证 |
| JSON/atlas/MOC 编码未统一 | 同语义在平台或库升级后字节与 cache digest 漂移 | JSON 用 RFC 8785 JCS，atlas 固定 ASCII/LF/order，MOC3 固定 attested endian/float/section codec；版本进入 stage fingerprint |
| component cleanup 与 Delaunay/采样仍依赖遍历顺序或随机 joggle | A 的容量计数与 B mesh 可分叉，两个 runtime 的骨架/权重也随环境漂移 | A `MaskComponentPlan` 单次冻结 labels/IDs；B `MeshBuildPlan` 固定量化、seed/symbolic perturbation、canonical vertex/triangle order 与 geometry dependency/options fingerprint |
| override target fingerprint 含义不明 | 自摘要循环，或算法/config 小改就丢失人工校正 | `target_input_fingerprint` 只绑定 see-through payload/canvas/tag schema；override SHA 单独进入 stage fingerprint |
| padding/extrude packed footprint 未定义 | A dry-run 与 C 实际装箱相差 4px/边，UV 可能采到 ring | content + 2px extrusion + 2px gap，footprint 固定 `w+8/h+8`；装箱用 footprint、UV 只指 content |
| `tblr_split=false` 或腿未拆分 | 不能安全生成双侧可动骨骼 | 不改默认值；merged/partial 状态 + 明确失败/刚性降级 |
| 背面/侧面角色 | source `-l/-r` 不等于可靠解剖侧 | 内部 `xmin/xmax`；`anatomical_side` 独立可空 |
| Rig schema 演进 | 旧 rig/override 无法读取 | `schema_version` + 集中迁移；首版未发布前不承诺 v0 |
| SDPose 约 0.95B / 官方包数 GB | 启动、显存、依赖冲突、批量吞吐风险 | 独立 worker/extra、默认关闭、Body 17 优先、阶段结束释放 |
| RTMW 包 229 MB | fallback 仍有缓存和 provider 成本 | 独立 ONNX extra、默认关闭、固定 graph 契约 |
| pose 权重和训练数据许可 | 不能稳定分发、镜像或商用 | 每 provider coherence group + SHA-256 + 基础模型/数据许可清单 |
| 静态 PSD 没有新表情纹理 | procedural blink/smile 可能视觉失真 | 默认 core profile 把表情设为 optional；分层 blink validator；严格 avatar profile fail fast |
| 共享 texture pack 超出 4×2048² | 两种格式无法满足冻结的资源预算 | A 阶段精确 dry-run fail fast，C 冻结同一 plan；不降采样、不回退散页；以后通过 profile/version 显式提高预算 |
| D/E 独立 PNG encoder 却要求字节相同 | zlib/filter/chunk/metadata 或依赖升级使同像素输出不同字节 | C 只编码一次 canonical PNG；D/E byte copy，raw/encoded 双 SHA 闭环，encoder 配置只进入 C fingerprint |
| 共享 PNG 被误解为共享最终 UV | Spine/Cubism 对同一 rect 发生上下翻转、偏移或错采样 | C 只存 top-left canonical UV；D/E 使用独立 attested adapter，四角/中心异色的非对称 runtime fixture 验证 |
| shared PNG 没有 straight/PMA 契约 | 半透明边在不同 runtime 黑化、发光或出现色边 | C 固定 straight-alpha sRGB bytes；Spine `pma:false`，Cubism target loader contract 进入 report/release matrix，半透明异色 fixture 实载 |
| A/B/E 对 texture region 粒度理解不同 | dry-run 与最终 plan 不一致，或稀疏 part 被偷偷重裁 | v1 固定一 canonical part 一 region；所有 component 与未来 skinning ArtMesh 共享它，分量级 packing 留给新 schema |
| Spine major/minor 不匹配 | runtime 直接拒绝数据 | 首版固定 4.2；其他版本明确不支持 |
| optional preset 被误当成必须双格式交集 | Spine 原生能力被无谓删除，或目录扫描得到错误能力结论 | `optional_preset_parity=per_format`；公共 manifest 逐格式状态/artifact 闭环，required 仍严格对称 |
| 骨骼权重不能直接进入 Live2D runtime | 文件能加载但独立关节弯曲缺失 | Live2D v1 不转换多骨 LBS，`wave.*` 仅 Live2D omitted；Spine 正常导出，未来 Live2D 先过 Glue gate |
| 完整 Rig bone 树被复制为 Live2D deformer | 大量无 driver section 增加引用和 consistency 风险 | `Live2DDriverLiveness` 剪枝；静态 rest transform 折叠，emitted/pruned/reparent 表与双向可达性 validator |
| 单一 PPU 公式被套到嵌套 deformer | 子层缩到原点、pivot 漂移或 warp 内 ArtMesh 尺度错误 | 逐节点 `Live2DCoordinatePlan`；E0-core 覆盖 root/warp/rotation frame 与 forward/inverse 0.1 px parity |
| E0 frame 结论没有绑定语义契约 | frame codec/layout/stack 顺序变化后继续复用旧假设 | `live2d_frame_contract_digest` 只钉专用 semantic-kernel 源码、descriptor、vectors 与已实测 Core allowlist；完整 compiler/writer hash 只作 provenance/cache |
| production driver/rank 表被混入 frame attestation | 新增纯内容 driver 也要求重新取得 Core 环境并重签 E0 | attestation 只证明 comparator、parent chain 与 runtime 组合语义；`RigidDriverRegistry` 行内容只进入 motion/export fingerprints |
| rigid-driver rank 跨语义域重复 | preset 演进后两个 required driver 落到同一 bone，默认 profile 才在生产时冲突 | built-in rank 全局唯一且 registry load fail fast；当前 v1 固定为 `100/200/300/400` |
| registry startup 错误被塞进 item diagnostics | 产生虚假 part/joint ID，`continue_on_error` 对程序配置错误继续跑 | draw/control/preset/profile/diagnostic/primitive/rigid registries 在 CI 与 job startup 按独立 code 校验；失败无 item staging/report，binding-plan 冲突才使用 item code |
| 同 bone 多个 rigid parameter 的 deformer 身份/顺序不定 | 两个实现都字节稳定却组合出不同顶点 | instance key=`(bone_id, parameter_id)`；`rotation-stack-v1` 显式 rank 外→内；非交换 synthetic E0 fixture |
| candidate enumerator 与 binding selector 漂移 | 罕见 Rig 在生产批次才以 `missing_export_symbol` 失败 | C 物化 immutable candidate universe，E 只能按 `candidate_id` 选取；generated Rig × 全 registry/profile 属性测试证明 superset |
| 同一 RotationDeformer 的 angle/origin 组合顺序错误 | pivot 平移时顶点走错路径 | 固定 parent-local `T(origin)·R(theta)·S(scale)` 语义；E0-core 同参数同时变化并取 9 点实载 |
| parameter default 插值不等于 rest | 模型加载后未播动作就歪斜且 consistency 仍通过 | default-rest invariant；必要时显式 default keyform，首次 update 后顶点/opacity/draw order 实载 |
| 无 Glue blend band 与刚性边界使用不同插值 | stop 间原理性裂缝/重叠 | 移出正式 v1；E0-S 只测量，未来 required joint bend 必须先验证 Glue 或另立可量化遮盖契约 |
| StretchyStudio runtime 实现缺 rotation hierarchy/expressions | 直接移植会产出“可打开但不满足需求”的包 | 只作 section/warp 参考；E 阶段补 RotationDeformer、parent binding、逐层坐标 plan 与 exp3 |
| MOC3 是未公开二进制格式 | 版本、Core 或 writer 变化会导致 consistency/load 失败 | 固定 V4.00 和参考 commit；parser + Core consistency + 实载三层 release gate |
| Cubism Core/SDK 的分发许可 | CI 或最终产品可能不能合法捆绑 validator/runtime | writer 与官方 validator 解耦；发布前完成许可证审计，不自动打包 Core |
| 多个 non-rigid 参数直接修改同一 ArtMesh，且只做逐 preset 检查 | 每个 preset 单独合法，合进一个模型后却要求多参数 grid 或产生非预期组合 | deformer 层级组合不算冲突；`FormatPresetSetPlan` 对完整 primitive union 做 required-first/固定 optional priority，无法拆分的 target 在 C 失败或 omitted |
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
   是其子集。NativeVariant 目录缺失必须得到固定 empty-set set/eligibility digests；合法且 A-eligible 的
   hidden variant manifest/PNG 加入独立 render part、mesh/atlas/anchor bundle但不改变 joint observations；
   合法但 quality-rejected 的 entry 只进 diagnostics/set digest，不占 region。绝对/越界/`..`/symlink path、
   非 canonical/reserved ID、占用或碰撞 `part/native.` derived namespace、未声明额外文件、SHA/尺寸/composite-mode 错误、重复 ID/role、未知 role、
   空/重复/跨 family base、base 列表不是 registry-derived expected set、引用不存在 base/anchor、anchor 不属于 base
   或不是 base 中最前层均必须 `input_contract_mismatch`，不能回退 procedural。`eye_closed.coupled` 正例必须
   稳定拆成两个 x-ordered component branch；少于/多于两个或永久共用一个 parameter 的 mutation 必须使
   native implementation 不 eligible，在有 procedural 时原子回退、core 无替代时 omitted、strict 无替代时
   `missing_required_capability`，不能误报成输入 schema 错误。
3. **双 `info.json`**：传入根 Marigold `info.json` 必须得到 `postprocess_info_required`，不能返回
   空 geometry；final part 缺 `xyxy/tag/depth_median` 任一字段均失败。
4. **canvas 契约**：768/1024/1280 fixture 与 `src_img`、manifest、PSD 一致；2048 或其他非白名单
   edge 必须得到 `unsupported_auto_rig_canvas_resolution`，伪造非方形 `frame_size` 必须得到
   `unsupported_non_square_frame`。测试还要断言 `SEE_THROUGH_PROFILES` 的 resolution 唯一集合精确
   等于 auto-rig 白名单 `{768,1024,1280}`；上游新增档位时必须显式适配。不写一个无法证伪的
   通用 H/W 顺序测试。
5. **分辨率门槛**：按 768/1024/1280、极端原图长宽比、小主体和细部宽度分桶，报告 joint
   resolved 率、轮廓误差和刚性降级率；A 阶段必须先定义并达到阈值。
6. **Rig 往返**：内存 Rig → JSON → load 后语义相等，所有 ID、capability、`ControlSpec`、
   `ControlBinding`、clip、
   expression、`FormatModelPlan`、`PrimitiveCandidateSet`、`GlobalExportSymbolTable` 和 texture-page
   引用有效；交换 exporter 顺序或
   改变 per-format pruning 后，完整 candidate/symbol-table digest 与共同 typed keys 的
   `{namespace_key, export_name}` 不变。另做 registry-driven 属性测试：生成覆盖缺失/merged/split bones、同一 preset 跨 torso/head
   control bindings 和可选 expression targets 的 Rig 集合，遍历测试运行时发现的全部
   profile/preset/control/parameter/
    primitive registry entries，断言每次 `binding_plan_candidate_ids ⊆ candidate_universe_ids` 且
     `binding_plan_keys ⊆ candidate_universe_keys`。测试不得硬编码当前四个 preset；新增 registry row
    必须自动进入矩阵。故意让 selector 返回未知 candidate 的 mutation fixture 必须在 writer 前失败。
    `InternalIdCodec/ExportNameCodec/SymbolKindCodec` 另使用独立 golden matrix：`part/front-hair → front_hair`、
    `clip/wave.xmin → wave_xmin`、`(bone/torso,parameter/auto_idle) → torso__rot_auto_idle`、
    `parameter/angle_x → ParamAngleX` 必须逐字节相等；单 component 的 `topwear` direct/component symbols
    在各自 namespace 可使用裸名，多 component 时 slot/attachment/ArtMesh 必须使用按完整 component internal ID
    算出的 `topwear__c_<16hex>`，而 atlas region 仍为 `topwear` 并由显式 path 命中。`part/a-b` 与
    `part/a.b` 在同一 namespace 的碰撞 fixture 必须让**双方**都得到 `truncate(preferred,46)` prefix + typed-key hash suffix，
    不能让遍历中的第一项保留裸名；同样裸名位于不同 namespace 则不得加 suffix。还要覆盖 63/64-byte
    边界、reserved parameter 冲突、identity-derived 64-hex internal slug、残余 suffix collision 注入，以及
    exporter 再次 sanitize/按存活集合重命名的负例。所有 artifact path/fragment 必须从对应 symbol 解析，
    `wave.xmin.motion3.json` 或 `#animations/wave.xmin` 这种直接使用 internal slug 的 mutation 必须失败。
    通过 test-only digest stub 让两个不同 identity records 命中同一完整 internal ID 时必须稳定返回
    `internal_id_collision`；不得改用 label/遍历序号继续。
    `motion_manifest.json` 还必须由落盘 Rig 确定性重算：单独篡改 supported/omitted、artifact 或默认 clip，
    即使 JSON schema 合法，也必须在 D/E binding 前因 `rig_json_sha256`/语义 projection 不一致失败；D/E
    不得因该篡改改变从 Rig 选出的 binding plan。
    draw-order fixture 还要覆盖远近深度、全部 depth 相等、semantic edge 与 depth 刻意冲突、未约束
    ready-set ties 和同 part 多 component；
    A 的 part rank 与 B 的 component expansion 必须分别命中自己的 stage，component rank 全局唯一且同
    part 连续，`max=min` 不除零，semantic edge 必须胜过 depth，改变 PSD/JSON/connected-component 遍历
    顺序不得改变结果。built-in DAG 必须无环；注入一条闭环边时 startup 以
    `invalid_draw_order_registry` 失败且不枚举 item。fixture 还必须把 eyewear 的 depth 刻意放到 face/eye
    后方，断言 `{face,eyewhite,irides,eyelash,eyebrow}<eyewear` 仍胜过 depth；删掉任一 built-in eyewear edge
    的 registry mutation 必须失败。`eyewhite/irides/eyelash` 同时拆成 xmin/xmax 的 fixture
    必须按 base tag 把每条 semantic edge 展开到实际 Part ID；缺一个 base tag 时不生成 phantom node，
    伪造 suffix 绕过语义边的 mutation 必须失败。多层 eye base + 一个最前 draw anchor + 两个 hidden
    variants 的 fixture 必须保持所有普通 base 的相对顺序，再对包含 anchor/variants 的 final sequence 从 0
    连续重编号；把旧整数 rank 强留给 base、产生碰撞/空洞，或只给 variant 使用分数 rank 都必须失败。把其余
    base 强挪进 bundle、选择非最前 anchor，或把无关 eyelash/face part 插入 anchor bundle 内的 mutation
    必须失败。
    全身、半身和 head-only Rig 都必须恰有一个 `bone/root` synthetic identity；head-only 不得因 pelvis/spine
    缺失而丢 root，任何非 root bone 的空 head/tail 必须被 validator 拒绝。
7. **缓存失效与单写者**：修改 manifest、mask、info、payload、tag registry、相关配置、preset 版本或
   overrides，精确失效相应 stage；只改日志级别不失效。完整跑一次后原样重跑，A/B/C 都必须命中
   cache 且所有 output SHA 不变；故意损坏 `rig.json` 的 C-owned 字段时只失效 C→D/E→G，B 的
   `RigGeometryCache`、manifest、mesh/weight SHA 均不变，且测试 spy 证明 B compute/writer 调用次数为
   零；损坏 B cache 才失效 B 及下游。
    同一 fixtures 在不同 `PYTHONHASHSEED`、Windows/Linux newline 环境与受支持 Python patch 版本下，
    所有 JCS JSON、atlas 和 MOC golden bytes 必须相同；JSON 尾随换行/pretty print、SHA-256 改成大写/
    base64/裸 hex、atlas CRLF/不同 page order、非 canonical float 或未清零 binary padding 的 mutation 必须改变 validator 结论而不是被 cache
    当成等价文件。
    修改 MaxRects/rectangle-builder/TexturePagePlan 任一语义时必须从 A 开始重跑并在 C 得到一致 plan；
    只改 `CanonicalPngEncoder`/Pillow/zlib 设置时 A/B 命中 cache、从 C 开始重跑。
    只在 `spine_4_2_dev`、core、avatar profile 间切换时，A/B 与 NativeVariant admission 必须继续命中，
    从 C 的 format/preset decisions 开始失效；只把 validation tier 从 structural 切到 release 时，C 也必须
    命中，只重跑需要目标 runtime gate 的 D/E 与 G。把整个 profile/config 文件 hash 塞进 A/B 造成无谓
    重算，或让同名 profile 暗改 A resource envelope 后仍命中，两个 mutation 都必须失败。
    新增、删除或修改 NativeVariant manifest/PNG 必须保持 base `target_input_fingerprint` 不变、使已有
    joint override 仍可应用，但 `native_variant_set_sha256` 改变并从 A 开始失效 draw/texture/mesh 及下游；
    把 variant digest 偷并入 target fingerprint 或只从 C 失效的 mutation 必须失败。修改 native role、
    coverage/role-envelope/occlusion-intrusion、side/component selector 或 anchor-expansion policy 也必须从 A 开始失效并重算 B/C；
    分别删除 required manifest、写坏 JSON、放入 escape symlink 的 early-A fixtures 都必须由 G 产出
    `error.json`：`observed_input_set_sha256` 非空且重复运行稳定，尚未形成的
    `target_input_fingerprint/native_variant_set_sha256/native_variant_eligibility_sha256` 为 `null`。修复对应文件必须改变 observed digest；
    inventory 若跟随 symlink、扫描 `rig/**` 或用 observed digest 冒充 canonical digest，mutation 必须失败；
    修改 `ControlRegistry` domain/parameter ID、`ControlBinding`/TargetTransfer schema、FormatModelPlan 或
    FormatPresetSetPlan kernel 时 A/B
    命中 cache、C 及对应 exporter/G 失效；只改 rigid stack rank 还必须使 C/E 失效但不得改变 frame
    attestation digest。修改 `MaskComponentPlan` threshold/connectivity/morphology/label dependency 必须从 A
    开始失效；修改仅属于 `MeshBuildPlan` 的 contour/sampling/triangulation/Qhull options 必须精确从 B 开始
    失效，不能重跑 A，也不能让旧 B cache 命中。修改 `InternalIdCodec` 或
    `mask-component-id-v1` identity schema 必须从 A 失效；只改 `ExportNameCodec/SymbolKindCodec` 时 A/B
    继续命中、从 C 重建 symbol/artifact plan。
    另断言所有 stage payload output path 两两不相交、manifest 不自哈希且非终态 manifest 的 SHA 被
    直接下游记录、B
   完成时公共 `rig.json` 尚不存在、部分 RigDocument 被
   D/E 拒绝；把 `rig.json` 同时放进 B/C manifest 的 mutation fixture 必须得到
   `stage_artifact_ownership_conflict`。completed fixture 保留旧 `export_manifest.json` 后修改 D encoder
   fingerprint，`skip_completed` 必须为 false，只重跑 D/G 而复用 E；仅删除 G manifest 时复用 A-E、
   只重跑 G。只有 export manifest、篡改任一 D/E artifact、错误 C/D/E manifest SHA 或旧 validator
    fingerprint 都不得 skip，并必须在执行最早失效 stage 前撤销旧 terminal artifacts。
    `completed_with_degradation` 在整条 DAG 仍有效时可以 skip；dev/structural 传 `--skip_completed` 必须在
    枚举 item 前拒绝，成功时只保留活动 stage report、不得生成 G manifest 或两种公共 terminal artifact。
    把 C page 数从 4 降到 1、把 E motion/expression 从 supported 改为 omitted 后，旧 page/motion3/exp3
    文件必须消失；D 重建的 `skeleton.json` 也不得保留已 omitted animation key。在 owner public
    namespace 手工塞入未列文件必须使
    inventory validator 失败。模拟发布中途崩溃时旧 stage manifest 已失效，partial output 不得被复用。
    A/B/C/D/E 任一失败还必须先撤销该 stage 的旧 success manifest、写非复用 `failure.json`，再由 G
    生成公共 error；把 `stage_failed` 写进 success manifest 或让 resume 接受 failure record 的 mutation
    必须失败。下一次成功提交删除本 stage 陈旧 failure record，但不得让别的 stage 共写该路径。
8. **左右约定**：不对称 fixture 断言 source `-r` 对应较小 x，内部只产生 xmin/xmax；
   `front hair` 必须得到 `base_tag="front hair"` 与 `part/front-hair`，`handwear-r/-l` 以及从未分侧
   handwear 两可靠组件提升的结果都必须分别得到 `part/handwear.xmin` / `part/handwear.xmax`；非 LR family 的多组件
   仍只有一个 base Part ID。registry slug 碰撞、数字后缀兜底或把 source `-r/-l` 原样带入 internal ID
   的 mutation 必须失败。
   `tblr_split=true` 同时覆盖成功替换和保留 base tag 两条路径。v1 的 xmin/xmax 眼眉 control 必须导出
   custom `Param*XMin/XMax`，并可被 EyeBlink group 正确驱动；未经版本化 anatomical-side observation/
   override 却输出 `ParamEyeL/ROpen` 的 mutation 必须失败。
9. **几何关节**：覆盖直肢、弯肢、袖口毛刺、断裂 mask、交叉/粘连，并逐项覆盖
   head_base/head_top/wrist/hand_tip/ankle/toe observation；resolved 关节在合法 mask 区域，错误
   场景产生 unresolved 而非假坐标。geometry-only wrist false-resolve `≤5%` 且不设 recall 下限；
   eligible ankle conditional resolved `≥80%`、false-resolve `≤5%`。
10. **网格/权重**：不跨连通分量，triangle 索引合法且面积大于零；每顶点 1-4 个有效
    influence 且和为 1；分辨率缩放后弧长权重分布近似不变。固定 mask/part 在打乱 contour、component、
    sample 和 Qhull simplex 输入顺序后，A `MaskComponentPlan` 必须先产生完全相同的 component
    IDs/label bytes/side/count，B `MeshBuildPlan` 再产生完全相同的量化 vertices/triangles/hull bytes；共圆、
    近共线、重复点和两连通分量 fixture 分别覆盖 symbolic perturbation 与失败路径。B 尝试重新 threshold、
    merge/split 或生成 A 表外 component ID 的 mutation 必须在 B validator 失败。`CanonicalLabelMap` 必须
    round-trip；改 magic/大小端/width/height、插入 trailing byte、label order 与 record 不一致或 hash 文件名
    不匹配都失败，test spy 证明 B 的 threshold/connected-component 调用次数为零。
    启用随机 `QJ`、直接保存 simplex 顺序、改量化/epsilon 而不升 plan version 的 mutation 必须失败；依赖
    升级只允许在 mesh golden 全量重签后进入 lock。
11. **纹理页计划**：A 对每 canonical part 的真实 padded bbox 做 dry-run，C 用同一 packer/排序
    重算出的 plan 必须逐字段一致；B 的连通分量以及 E0-S/未来 Glue 的 skinning ArtMesh 拆分不得改变 region
    数量/rect。对每个 `w×h` content 必须断言 `content_rect ⊂ extrusion_rect ⊂ packed_footprint`、
    footprint 尺寸精确为 `(w+8)×(h+8)`、2px ring 是最近边缘 straight RGBA、外侧 2px gap 为透明；
    MaxRects overlap/budget 使用 footprint，而 UV/atlas lookup 只使用 content rect。另断言 2048²、最多
    四页、used index 精确为 `0..N-1` 且路径只用无前导零的 `page_<index>.png`、no-rotation、无 packed-footprint overlap、canonical
    `page_top_left_v_down` UV/rect
    闭环、`rgba_sha256` 与 `encoded_png_sha256` 各自正确。测试 spy 必须证明 C encoder 每个 used page
    恰好调用一次、D/E image encoder 调用次数为零；两 exporter 只做 byte copy，三份对应文件 SHA
    逐页完全相同。canonical PNG 必须是声明过的 straight-alpha sRGB bytes，不含 ICC/gAMA/tIME 等
    profile/非确定 metadata；Spine atlas 每页 `pma:false`，Live2D report/manifest 的 runtime loader contract
    与 release harness 一致。encoder 或 texture-contract version/settings 改变必须失效 C 及下游；
    分别核对
    `budget_occupancy=sum_area/(4×2048²)` 与
    `used_page_fill=sum_area/(used_pages×2048²)`。mandatory base 超预算必须在 A 返回
    `texture_budget_exceeded`，不能等 C、不能散页或静默缩图；native group rejection 另计数量/面积且不混入
    base failure rate。真实样本按 resolution/tblr_split
    统计两个比率、page count 和 MaxRects 成功率，不能用总面积估算替代实际装箱。
    构造“base 恰可装下、加入 blink 可装、加入 mouth-open 不可装、随后较小 mouth-form 又可装”的 fixture，
    A 必须每次从空 plan 重算、原子接纳 blink/拒绝 mouth-open/继续尝试并接纳 mouth-form，最终 core item
    不失败；把失败 group 拆开塞入、拒绝后停止后续尝试或把 warning 升级为
    `texture_budget_exceeded` 的 mutations 都必须失败。
    另构造 base 999 components + 三 drawable 的 native group：A 必须以
    `native_variant_drawable_budget` 拒绝整个 group，C 对最终 base 仍可通过；base 自身 1002 components 则
    必须留给 C 以 `draw_order_capacity_exceeded` hard fail，不能误报 optional warning。
12. **动作/表情**：preset 同输入逐字节确定；默认 core profile 只要求四个不依赖新表情纹理的
    结构/骨骼动作。`ControlRegistry` 必须满足 ID 唯一、domain/default 合法且 standard/custom parameter
    无碰撞；非法 domain 或同一 parameter 两套 domain 在 startup 返回 `invalid_control_registry`。所有
    `ControlBinding` 的 binding ID 与 `(implementation,control,target,property)` tuple 必须一一对应、transfer
    digest 唯一且 default 回到 rest；每个 binding group 的 implementation rank 唯一，选择结果恰好一个
    完整 atomic bundle。用 blink fixture 同时提供 native/procedural：C 必须只选择 native；删掉 native bundle
    一项时必须整体回退 procedural，不能混搭。native variant **visibility-branch envelopes** 必须全在
    `[0,1]`、同 `binding_group_id+control_id` 在所有 stops 的和 `≤1+1e-6`、default 全为 0；普通 base setup opacity
    必须始终为 1，任何 native binding 指向 base opacity 都失败。coverage fixture 必须同时命中
    `coverage_leak≤0.01`、role 对应 `alpha_mass_ratio≤k_role` 与 `occlusion_intrusion≤0.01`；透明闭眼线稿、覆盖整张脸、把 mouth-open 与 mouth-form 都写到
    base opacity 的 mutations 均不得被选为 native。A/B 的 draw/component ranks 只对 anchor + variants 连续，其他 base 保持原顺序，
    另用 `mass_B=200,mass_V=2000` 的共同尺度 fixture 验证
    `round_half_up(0.20×sqrt(200))=3 px`、`round_half_up(1.00×sqrt(200))=14 px`、
    `round_half_up(2.00×sqrt(200))=28 px`；mouth-open 只在该 28 px base-anchored envelope 内扩张的合法
    patch 可以通过，旧 `min(mass_V,mass_B)` 分母 mutation 必须失败。构造 `mass_V=11×mass_B`、不接触
    nose/eyebrow、但至少 `0.02×mass_V` alpha 落在 28 px envelope 外且覆盖可见 face 的 cheek patch，必须命中
    `occlusion_intrusion`；把授权域偷换为 `support(a_V)` 的 mutation 会错误放行，因此 golden 必须拒绝该
    实现。`mass_V/mass_B>12` 的整脸 patch 必须命中 `alpha_mass_ratio_exceeded`，合法 ratio 内覆盖
    nose/eyebrow 的 patch 也必须命中 `occlusion_intrusion`。再构造 alpha mass 同为 200、cleaned bbox span
    分别为 50/200 px 的两份 base，三种 role radius 必须逐项相同；任何恢复 `support_span_B`、允许 role 覆盖
    共同 scale kind、缺 radius、写 `null` 或从 variant support/bbox 推导 radius 的 mutation，都必须在 startup
    命中 `invalid_primitive_registry`。
    intrusion 诊断至少覆盖 face-only、nose-only 与 mixed 三例：逐 Part ratio 必须按 stable `part_id` 排序，
    与 ordinary prefix part 集合精确相等并保留 0 contribution；face/other aggregates 分别求和，二者与逐 Part
    总和都必须闭合到 `occlusion_intrusion`。只写旧
    `face_alpha_outside_envelope`、截断 top-N、漏掉 nose contribution 或按输入遍历顺序输出的 mutation 必须失败。
    blink fixture 要覆盖 ratio `1.25` 通过、`>1.5` 拒绝，以及相同 `mass_B` 放在不同 canvas resolution 时得到
    相同 radius；缩放 feature mass 时 radius 按平方根变化并分别命中 `2/12 px` clamp，任何恢复
    `canvas_edge/128` 的 mutation 必须失败。coupled 双眼必须逐 component 得到与两张单侧 fixture 相同的
    两个 radius/判决；先合并 mass 再给两侧共用 `sqrt(total)` 的 mutation 也必须失败。
    Spine/Live2D setup opacity 和半透明中点 composite 必须一致。把 transfer 塞回 clip/expression、复制同一 tuple 或让
    expression 引用无 binding 的 control 必须在 C 失败。所有
    MotionClip 必须是 30 Hz、整数 frame、linear segment，loop 的首尾 control 值严格闭合；同一 clip/
    control 只能有一条 curve，同一 parameter 不能被两个 control curve 重复写。用一个 idle control 同时
    驱动 torso rotation 与 head translation 的 fixture，必须通过两条顶层 binding 的明确 transfer 得到 degree/pixel 输出；
    把 target property keys 直接当 parameter keys 或制造两条 ParamAutoIdle curve 的 mutation 必须
    `invalid_motion_clip`。所有 transfer 在 control default 必须回到 canonical rest；故意把 scale default
    写成 0 或让 sampled deform default 改动顶点时必须在 C 失败。Spine/Live2D 在每个 frame 及相邻 frame 中点求值，与 Rig canonical
    control+transfer evaluator
    的 property/landmark 误差都在阈值内。canonical transform fixture 的 `+10 px y` 与 `+15° clockwise`
    必须在 Spine 编码为 Y/angle 负号，在 Live2D 经 Core/runtime inverse 后仍回到同一向下/顺时针 canvas
    轨迹；两边 harness 固定 single clip、weight=1、zero fade/mix。故意使用格式默认 Bezier/stepped、
    Live2D 非零 FadeIn/FadeOut、Spine 非零 AnimationState mix、漏做轴反射或 exporter 重采样必须失败。
    `motion-core-v1` golden 必须逐项等于本文 descriptor 的 kind/duration/loop/control keys/transfer 公式；
    任改一个 frame、幅度或 side sign 而不升版本都失败。`blink/talk` 必须只出现在 clip/motion artifact
    namespace，`happy/sad/surprised` 必须只出现在 expression namespace；把 blink 写成 exp3 或 happy 写成
    循环 MotionClip 的 mutation 必须在 C/projector 阶段失败。
    `head_nod` 与 `head_shake` 必须各自产生非空、可区分的 landmark 轨迹，并明确
    标记为 2D stylized motion，不能以标准参数 ID 冒充 3D 新视角；merged
    limb 不产生 wave；完整 limb 的正式 dual-runtime fixture 必须在 Spine 生成 wave、在 Live2D 以
    `live2d_joint_bend_requires_glue` 省略，并与 manifest 的 `supported_formats=["spine_4_2"]`、artifact
    引用逐项一致；layered blink 分别验证眼白/虹膜残留和睫毛厚度；strict avatar 缺 required
    capability 时失败；`dual_runtime_avatar_v1` 的 required 集合必须精确包含 blink/talk/happy/sad/surprised，
    不能由配置把 procedural/native 门槛改成第三种含义。两个 exporter 消费同一份 preset 语义和逐格式决策，不能各自猜测。
    每个 `(preset,format)` 还要断言 C preflight 与 D/E recompute 的 planner input/output digest 相等，并对
    每个 format 断言 `FormatPresetSetPlan` 的 required/optional 输入、priority、primitive union 与最终摘要相等：
    optional 超出 non-rigid 预算时必须在 C 写 omitted 且 exporter writer 不被调用，required 同条件在 C
    失败；故意让 E 改 stop planner、把 supported 临时降为 omitted 或返回不同 plan digest，必须
    `format_plan_mismatch` 失败而不是修改 Rig/projection。构造 `mouth_open` 与 `mouth_form` 各自单独可编译、
    合并后争用同一不可拆 ArtMesh 的 fixture：core Live2D 必须按固定 priority 保留 talk、omitted happy/sad
    并记录冲突，Spine 可按自身 set plan 保留；avatar profile 因 required-required 冲突必须在 C 失败，E
    writer 调用为零。交换 registry/dict 遍历顺序不得改变选择结果。dual-runtime expression fixture 只接受
    `overwrite_full_weight`：在同一 base motion 后，Spine expression 的每个 target 必须只有
    `time=0,1/30` 两个相同 key，并按 manifest 的 track-1/loop/replace/full-alpha/zero-mix hold contract 应用；
    零时长单 key、任意延长 duration 或依赖 completed entry 残留的 mutation 必须失败。Live2D 按 zero-fade
    Overwrite exp3 应用，语义 landmarks 必须在容差内；把 Add/Multiply 塞进 dual
    profile 必须在 C 失败。同一 control 的 motion+expression 正例必须由 `suppresses_controls` 明示且最终
    值以 expression 为准；expression 在两个 runtime 中都保持到显式 clear，clear 后下一 update 必须恢复
    base motion 当前值。每个 supported expression 与 idle 的 compatibility matrix 必须为 compatible 并实载；
    native-overlay fixture 的 `talk+happy/sad` 必须因 component-level coverage overlap 逐格式列入
    `incompatible_with`，同时存在不同 variant ArtMesh/文件也不能宣称可叠加；xmin/xmax overlays 即使引用
    同一个未拆 Part ID，只要 A partition 的 actual component support 不交叠就必须保持 compatible。两个
    不同 control 争用不可拆 non-rigid target 的负例才应使模型级 preflight 失败/optional omitted。
13. **Spine 结构**：weighted/unweighted/animated golden fixtures 断言 flat arrays、bone indices、
    multi-page/multi-region atlas、effective attachment paths、namespaced global symbol table 子集/摘要、
    坐标转换和 animation timeline。每个连续 Spine timeline segment 必须省略 `curve` 以表示 linear；
    写入 `curve:"linear"`、`"stepped"` 或 Bezier 的 mutation 都必须失败，并在 runtime 中间点再次与
    canonical evaluator 比对，不能只做 JSON shape 检查。`SpineCoordinatePlan` 必须把 canvas 四角/中心、所有 bone joints 和
    weighted bind positions 按 `(x-W/2,H/2-y)` 转换并以 `≤1e-6 px` round-trip，synthetic root 恰为
    `(0,0)`；只翻 Y 或使用 bottom-left origin 的 mutation 必须失败。另用至少三级、每级非零 rotation/
    translation 的 hierarchy，按 parser 重建 `W_b` 后逐 bone head/tail 与逐 influence weighted setup vertex
    回到 canvas 都须 `≤0.1px`；直接减 parent head、只减 angle、或对所有 influence 复用 slot-bone local
    point 的 mutation 必须失败。一个 part 含两个断开的 component 时必须产生两个连续 slot、两个
    同时可见的 setup attachment，并命中同一个 part atlas region；slot bone 是 influence bones 的最近公共
    祖先，但 weighted vertex indices/weights 保持不变。把两个 component attachments 塞进同一 setup slot
    的 mutation 必须因少显示一个 component 失败。同一 `topwear` 派生出的 Spine slot/attachment-key/atlas-region 与
    Live2D Part/ArtMesh 允许保持同名；两个 skin/slot scope 内的 attachment key 也允许同名，但 actual
    attachment name 仍在自己的 skeleton-global namespace 验证，必要时显式写不同 actual name/path；同一
    namespace 内两个不同 typed key 同名则必须 `export_name_collision`。省略 `path` 和显式 `path`
    两条 fixture 都要恰好命中一个 region；exporter 内故意再次 sanitize、改变 namespace 或按存活集合
    消解名称的 fixture 必须失败。另用四角/中心异色且 mesh 不对称的 atlas fixture，让
    `Spine42UvAdapter` 经 4.2 golden/parser 后在目标 runtime 实际采样正确 landmark；半透明标记还要
    证明 `pma:false` 没有黑边/发光。把 `v_top` 原样写入、重复翻转 V 或把 atlas 改成 `pma:true` 的
    mutation 必须失败，不能只断言 UV 落在 `[0,1]`。用眼白/虹膜/睫毛/前后发和两块交叉肢体的
    非对称 overlap fixture，将 Spine 默认帧与按 Rig rank 做 straight-alpha source-over 的 canonical
    composite 比较；slot 反序或自行按 tag 排序必须失败。
14. **Spine 实载**：opt-in 4.2 Editor/runtime 加载并逐个播放 required clips，不能出现 recovery、
    missing region、schema error 或 attachment 跳变；canvas landmark 经 runtime world vertices 和 plan inverse
    回映后误差 `≤0.1 px`。发布 Spine 支持前必须真实执行一次。
15. **Live2D 普通 CI**：不依赖 Cubism Core 的 golden fixtures 断言 MOC3 header version 3、SOT
    offset/count、Rotation/WarpDeformer、parent_deformer_indices、ArtMesh index/UV/texture、parameter
    binding、rest/deformed keyforms、`CubismV400UvAdapter`、schema 和 deterministic bytes；每个 emitted
    envelope fixture 必须包含至少一个非 64-byte-aligned 的合法 body section offset，官方 V4.00
    Hiyori/Mark/Rice golden 也必须通过；把 `csmAlignofMoc=64` 重新误用为 blanket section alignment 的
    mutation 必须失败；每个 emitted
    parameter 的 ID/min/default/max 必须逐字段等于其 `ControlSpec.format_bindings.live2d` 与 domain，
    motion/expression 值不得越界，unused ControlSpec 不得形成 dead parameter；artifact basename 必须精确为
    `model`，从 item/display name 派生或出现其他 `.moc3/.model3.json/.cdi3.json` sibling 的 mutation 必须失败；model3 中实际声明的
    Moc/Textures/Motions/Expressions/Groups 全部可解析，core fixture 没有 supported expression 时允许
    Expressions section 与 exp3 文件同时缺席，但禁止空引用、孤儿文件或 manifest/filesystem 不一致。
    每个声明可见效果的标准/custom parameter
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
    否则得到 `live2d_dead_deformer`。剪枝前后的 default rest vertices 必须一致。同一 overlap fixture 的
    Cubism 默认帧必须与 canonical composite/Spine 在颜色容差内；draw-order 数值方向反转或按 section
    index 重排必须失败。1001 个 component 的边界 fixture 必须得到唯一 `0..1000` draw order 并通过；
    1002 个 component 必须在 C 的 `FormatModelPlan` 以 `draw_order_capacity_exceeded` 失败，E writer
    调用次数为零。截断、取模或让重复 draw order 依赖 Part/section 顺序的 mutation 均不得通过。
16. **Live2D deformer、坐标与 non-rigid 逼近**：E0-core 的
    `root→warp→rotation→rotation→ArtMesh` 在 rest、单层和同时驱动时，每个参数区间 9 点的 runtime
    顶点都与解析 frame stack 相差 `≤0.1 px`；全部 vertex/control point/origin 的 coordinate
    round-trip 同样 `≤0.1 px`，并显式断言 canvas top/bottom 经
    `(H/2-y)/PPU` 映射到正/负 root Y；旧 `(y-H/2)/PPU` mutation 必须失败。超阈值返回
    `live2d_coordinate_roundtrip_failed`。同一 RotationDeformer
    的 angle 与 origin 必须在一个参数上同时变化。所有参数设为 default、完成首次 model update 且
    尚未应用 motion/expression 时，vertices/rest transforms 必须 `≤0.1 px`，opacity/draw order 也与
    rest 相等；缺失显式 default key 导致插值偏离时返回 `live2d_default_pose_mismatch`。随后验证
    `T(origin)·R(theta)·S(scale)` 顺序。E0 的非对称五色纹理还必须证明 Core UV/parser/final render
    采样同一 page region，半透明点按声明的 target loader mode 得到预期 straight-alpha composite；任何
    U/V 轴翻转、region offset、边界取样、double premultiply 或 missing premultiply 都使 attestation 失败。
    `RigidDriverRegistry` load test 必须证明 built-in rank 全局
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
    每 binding 17-stop、按 ArtMesh vertices/warp control points 公式累计的 100 万 positions 和 64 MiB
    的失败路径；三个数字只断言 capacity guard，不生成质量分。让两个 preset 引用同一 payload digest
    的正例只计一次；同 binding 使用不同 payload 却试图去重的 mutation 必须失败。opacity-only binding
    不进入 positions 计数，但要在全区间满足 `≤1/255` scalar error、17 stops、variant envelope `sum≤1`/
    default=0 与 coverage/role-scale/occlusion-intrusion 门。
    E0-S 则必须观测并记录无 Glue 边界的弦割/seam，产生 experimental warning 而不是要求零缝或
    阻塞 E0-core。默认 core 四个 required preset 同时存在不得触发 `live2d_parameter_conflict`。
17. **Live2D expression 行为**：E0-core 在同一个有可见 keyform binding、不会 clamp 的参数上先播放
    motion；底层 API 在 full weight 下断言 Add/Multiply/Overwrite 为 `p+v`、`p*v`、`v`。端到端
    exp3 固定零 fade、推进至少一次 update 并确认 effective weight=1 后，才断言参数与
    landmark/像素变化；禁止用 manager 第一帧冒充 full weight。任一 blend 只改数值不改渲染、
    或更新顺序不符，均报 `live2d_expression_application_failed`。这里的 Add/Multiply 是 writer/runtime
    conformance 覆盖，不等于把它们升级成 dual-runtime product capability；正式双格式 expression 另按
    第 12 条只使用 Overwrite。
18. **Live2D startup、attestation 与 opt-in release gate**：structural/release tier 都先验证 packaged
    draw/control/preset/profile/parameter/diagnostic/primitive/rigid registries，再在枚举 item 前重算
    `live2d_frame_contract_digest` 并运行 attested pure vectors；registry 失败按类型使用 job-level
    `invalid_draw_order_registry`、`invalid_control_registry`、`invalid_preset_registry`、
    `invalid_profile_registry`、`invalid_diagnostic_registry`、`invalid_primitive_registry` 或
    `invalid_rigid_driver_registry`；缺文件或 frame/layout semantic-kernel、
    codec/transform/rotation-stack descriptor/vector 任一不匹配时返回
    `live2d_coordinate_schema_unverified`，
    不创建 staging。完整 compiler/writer provenance 改变只失效 export cache 并进入报告，不单独使
    coordinate gate 失败。普通 CI 消费现有 attestation、跑完整 structural pipeline，但不重跑 E0、
    不需要 Core，也不能写 completed manifest。release tier 在枚举 item 前还要求 Core version
    `≥04.02.0004`、实际导出 `csmHasMocConsistency`、binary SHA-256 位于 attested allowlist，且 SDK
    harness 的 validator protocol digest 一致；版本 floor/API probe 只负责提前拒绝，不会自动加入
    allowlist。缺失得到
    `live2d_release_gate_unavailable`，未 attested 得到
    `live2d_core_unattested`，不得由 worker 临时生成 attestation。startup 通过后，对每个正式输出执行
    `csmHasMocConsistency`，再用官方 SDK 或 Viewer 加载、渲染非空默认帧，驱动参数 extrema，并
    逐个播放 required motion/expression；另用最小 namespace fixture 让 Part 与 ArtMesh 使用相同裸 ID，
    证明官方 runtime 能分别索引两个 section。若实载不支持，必须升级 `namespace_schema_version` 并合并
    相应 collision scope，不能按 item 临时加后缀。该结论不属于 frame-contract attestation；
    attestation 也不能替代逐 item release validation。
19. **双格式原子交付与状态**：对同一 Rig，Spine 与 Live2D 的 required preset ID、
    `sample_rate_hz/duration_frames`（以及派生秒数）、linear interpolation、loop 和 capability 结论必须
    一致，语义 landmark 在全时间网格容差内；optional preset 允许格式不对称，
    但每个实际 animation/motion/expression 必须与 Rig format decisions 及其
    `motion_manifest.formats` projection 双向闭环，禁止未声明文件
    或声明 supported 却缺 artifact；D/E 重算的静态 `FormatModelPlan`、逐 preset plan 与完整
    `FormatPresetSetPlan` digest 还必须和
    Rig 完全相等；`export_manifest.json` 中 C/D/E manifest、三组 artifact-set、
    motion manifest/runtime-application contract、global symbol table、shared texture/UV/alpha/loader
    contract、target/native-variant-set/native-variant-eligibility/config/override digests、profile 和 validator
    fingerprints 必须逐项匹配当前磁盘与
    G expected fingerprint。
    无降级时写 `completed`；正式双格式
    `--allow-partial` 仅在两边验证通过且 required capability 仍满足时写
    `completed_with_degradation`。`spine_4_2_dev --allow-partial` 只写
    `stage_validated_with_degradation`。任一 A-E item failure 都不写 `export_manifest.json`，由 G 原子写
    公开 `error.json` 并返回对应错误；成功时 G 删除 error 后写 export manifest。两者 XOR，G manifest
    的 output SHA 必须匹配当前终态文件。A/B failure fixture 在删除全部 cache 后仍能从 item 根目录读取
    error 和 repair；D/E 并行同时失败时，`failed_stages/failure_records` 必须按 stage name 稳定聚合两条，
    不能“最后一个异常覆盖前一个”；job-startup failure 则不得伪造 per-item error。D 的 stage report
    可以保留，但不等于正式交付完成。模拟 G 原子替换/manifest 写入失败时，batch result 必须返回
    `terminal_finalization_failed`、不存在有效 G manifest，且任何 partial terminal artifact 都不能触发
    completed/skip；该错误不强求写入已证明不可写的 `error.json`。
20. **SDPose 契约**：真实 Body 模型组件摘要、`1024×768` 预处理、17 点/score 输出、timestep
    和坐标反变换固定测试；与官方脚本对同一 crop 做 parity 对比。
21. **RTMW 备选契约**：真实 ONNX graph metadata、SHA-256、预处理和输出 shape 固定测试；SDK
    `pipeline.json` 的错误 image_size 不得污染运行时。
22. **姿态价值评测**：建立人工标注动漫子集，按普通、弯肢、交叉、缺失分层，对比 geometry、
    geometry+SDPose、geometry+RTMW 的 normalized joint error/PCK、左右交换率、unresolved 率、
    吞吐/峰值显存和人工 override 数。仅“落在 mask 内”不足以证明正确；没有显著降低 override
    数时，两个模型都保持实验性并默认关闭。

阶段验收：A 先证明输入和几何质量；B 再做 mesh/weights；C 冻结 ControlSpec/ControlBinding/target transfers、common
preset、profile、FormatModelPlan/逐 preset/集合级 preflight、GlobalExportSymbolTable、TexturePagePlan 与 canonical
PNG bytes；
D 通过 Spine setup/animation 实载；E 通过 Live2D consistency、渲染、motion/expression 实载，
G 再验证完整 DAG 并发布成功/失败终态；F 只有在评测结果支持时才进入可选产品路径。

---

## 审查修订对照

| 审查发现 | Revision 2/3/4/5/6/7/8/9/10/11/12/13/14/15/16 处理 |
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
| 非刚性 keyform 没有误差/体积契约 | 对 warp、blink/talk 和实验 joint blend 使用 adaptive stops、逐顶点 5% 局部半径、每 unique binding 17 stops、按 emitted vertices/control-points 求和的 100 万 positions、64 MiB 和禁用多参数 grid；三个上限只作 capacity guard |
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
| A 执行 texture dry-run 但 packer 版本只失效 C | TexturePagePlan/rectangle/MaxRects 契约同时进入 A/C fingerprint；encoder/runtime 仍只从 C 失效 |
| G 对所有 item 恰好一次与 dev/structural 永不 completed 冲突 | G success 只属于正式 release；所有 profile 的 A-E failure 仍走 G，F 实验失败不改产品终态。非发布成功只产 stage report，禁用 `skip_completed` |
| 可变 page/motion/expression 目录没有精确 inventory | `output_file_sha256[]` 冻结为精确集合并加 inventory digest；旧 marker 先失效，obsolete owner 文件清完后才提交新 manifest |
| motion manifest 成为 Rig 之外的第二套 capability 事实 | 改为绑定 Rig file/semantic/symbol digests 的 C-owned deterministic projection；D/E 从 Rig 选 binding，只交叉验证 projection |
| C 冻结 capability 后 E 仍能临时 omit optional preset | C 运行共享的逐格式纯 preflight 并持久化 status/reason/plan digests；D/E 只能复算核对，late mismatch 或 validator failure 均使 item 失败 |
| exp3 Add/Multiply/Overwrite 被误当成 Spine/Live2D 共享表达式语义 | dual-runtime v1 收窄为 full-weight overwrite；Spine 用独立 animation + manifest application contract，Live2D 用 Overwrite exp3；其余 blend 只进 E0 |
| canonical page 相同被当成 Spine/Cubism 最终 UV 相同 | C 只冻结 top-left pixel/UV mapping；D/E 用各自版本化 adapter，Spine golden/runtime 与 Live2D E0 以非对称五色纹理验证最终采样 |
| shared texture 缺 alpha-mode/loader 契约 | canonical page 固定 straight-alpha sRGB；Spine atlas 写 `pma:false`，Cubism Native/Web loader mode 写入 report/manifest 并用半透明异色 E0/release fixture 验证 |
| MotionClip 只有 keys、没有 interpolation 语义 | v1 冻结 30 Hz rational-time integer frames 与显式 linear segments；D/E 不重采样，运行时在 frame/中点做全轨迹 parity |
| depth 被使用却没有 canonical draw-order policy | A 生成 back-to-front part rank（semantic precedence DAG + depth bucket + stable ID），B 展开连续 component rank；Spine/Cubism 只做方向适配并以 overlap composite fixture 验证 |
| 同 part 多 component 被错误映射为一个 Spine slot 的多个 setup attachment | 改为每 component 独立连续 slot/setup attachment并共享 part atlas region；weighted bones 不受 slot owner 简化 |
| `root` 依赖 pelvis→spine 导致半身/头像无根 | 新增永远存在的 synthetic identity `bone/root`；pelvis→spine 改为可选 lower_torso，所有 fallback 最终挂 root |
| Spine canvas 转换只有 Y 方向、没有原点/单位 | 固定 `SpineCoordinatePlan v1`：1 unit=1 px、canvas center origin、`(x-W/2,H/2-y)`，root 与 runtime landmark 双向验证 |
| Live2D root Y 方向与 canonical rotation 正号未冻结 | root 改为 `(x-W/2,H/2-y)/PPU`；Rig 固定视觉顺时针正，Spine/Live2D 都通过 runtime inverse 与同一 canvas evaluator 比较 |
| Spine linear timeline 被误写成显式字符串 | 按官方 4.2 JSON 契约以省略 `curve` 表示线性；`curve:"linear"`/stepped/Bezier mutation 必须失败 |
| Cubism draw order 的 `0..1000` 容量未进入 preflight | 增加 C-owned `FormatModelPlan`；1001 component 正例使用连续原值，1002 直接 `draw_order_capacity_exceeded` |
| semantic 遮挡关系只在 depth 同 bucket 才生效 | 改成 behind→front DAG + Kahn 拓扑排序，depth 只排序 ready set；规则按 canonical base tag 展开到 split Part ID，`head_core` 不再被误当 drawable part ID |
| MotionClip 把 control 值与 target property 值混为一层 | 引入 `ControlSpec/ControlCurve/ControlBinding/TargetTransfer v1`；binding 在 Rig 顶层单写，motion/expression 只驱动 control，Spine 求值 transfer，Live2D 每 parameter 只写一条 curve |
| image-space `xmin/xmax` 眼眉 control 被当成 anatomical L/R | v1 固定 custom XMin/XMax parameter 并允许 EyeBlink group 引用；标准 L/R 映射等待显式 side observation/override 与 schema 升级 |
| fixed preset 没有固定 duration/curve/transfer | `motion-core-v1` 增加可执行 descriptor 与 geometry-normalized 数学公式，所有内容改动必须升级版本并失效 C |
| blink/talk 的 MotionClip/ExpressionPreset 身份冲突 | 固定 blink/talk 为 motion，happy/sad/surprised 为 expression；manifest/projector 按 kind 校验 artifact namespace |
| motion fade/mix 没有共同运行时契约 | `MotionRuntimeApplication v1` 固定 Live2D 零 fade 与 Spine alpha1/mix0；下游自定义 crossfade 不纳入 single-clip parity 声明 |
| Spine parent-local/bind-local 数学未冻结 | 新增 `SpineBindPlan v1`，用 parent/bone world affine inverse 生成 local bone/逐 influence vertex，并以 rotated hierarchy 正向重建验证 |
| 逐字节确定性没有覆盖 JSON/atlas/MOC codec | 所有 JSON 采用 RFC 8785 JCS；atlas 固定 ASCII/LF/order；MOC codec 由 attested endian/float/section descriptor 钉住 |
| override fingerprint 可能自包含或被算法配置污染 | 字段改为 `target_input_fingerprint`，只绑定 see-through payload/canvas/tag schema；override file SHA 独立记录 |
| TexturePagePlan 未定义 padding/extrude footprint | 明确 content、2px extrusion ring、2px transparent safety gap；footprint=`w+8/h+8`，UV/atlas region 仅引用 content |
| preset 单独可编译但合并后发生模型级 non-rigid 冲突 | 新增 `FormatPresetSetPlan`；required-first、optional 固定 priority，对完整 primitive union 预检并持久化 set-plan digest |
| native/procedural binding 没有互斥所有权 | ControlBinding 以 semantic group + atomic implementation bundle 表达替代方案；C 固定 canonical/native/procedural rank并恰选一个，D/E 禁止混搭 |
| stage failure record 与成功 commit manifest 混用 | success manifest 只允许 `stage_validated*`；失败撤销旧 marker 后写非复用 stage-local failure，G 再发布公共 error |
| serializer 确定但 component/mesh topology 仍不确定 | A `MaskComponentPlan` 冻结 cleanup/labels/IDs，B `MeshBuildPlan` 固定采样/量化/symbolic perturbation/canonical topology 与各自 dependency fingerprint |
| packaged preset/profile/diagnostic/primitive registry 错误没有启动码 | 增加独立 `invalid_*_registry` job-level errors；枚举 item 前失败，不受 continue-on-error 影响 |
| 静态 PSD 无 native 表情层，却允许 override 任意指图 | 新增受限、item-local `NativeVariantSource v1` 与独立 semantic digest；路径/role/base/anchor 非法硬失败，目录缺失只表示无 native capability |
| native replacement 写 base opacity、role 尺度错误，或 optional variant 先挤爆公共资源 | 改为 coverage + role-scale + intrusion gated `occluding_overlay_v1`，普通 base opacity 恒 1；A 在 atlas 前做 atomic quality/admission，超纹理/drawable 预算只拒绝 optional group，最终 admitted set 才进入 B/C |
| early-A 错误尚无 canonical digest，且 G/F 失败边界含糊 | `ObservedInputInventory v1` 为 A-E terminal failure 提供 total evidence；canonical digest 可空，G 只发布 A-E production failure，F 实验 failure 不改产品终态 |
| 插入 variant 仍要求普通 base 数值 rank 不变 | 普通序列先冻结相对顺序，anchor token 展开后对 final sequence 连续重编号；碰撞、空洞与分数 rank 均非法 |
| profile/tier 无差别失效 A/B，Spine expression hold 又依赖 runtime 偶然行为 | stage-scoped relevant-config projection 使 profile 从 C、tier 从 D/E 失效；`SpineExpressionHold v1` 固定两点常值 loop timeline 与 clear 语义 |
| Live2D 输出路径仍使用未定义 `<model>` | v1 固定 artifact basename=`model`；三份入口文件路径进入 layout schema/inventory，禁止从 item/display name 临时清洗 |
| texture page index 的起点/格式未冻结 | 三份目录统一使用 0-based contiguous、无前导零 decimal basename；1-based/zero-padded/sparse mutation 由 plan/inventory validator 拒绝 |
| SHA-256 值的文本表示未冻结 | `DigestEncoding v1` 固定 JSON/API 为 `sha256:` + 64 lowercase hex，digest 文件名为无前缀 lowercase hex；大小写/base64/截断差异不再污染 JCS/cache identity |
| internal/export name 只声称稳定却未冻结 ID/name codec | `InternalIdCodec` 使用 typed prefix + semantic slug/完整 identity digest；`ExportNameCodec/SymbolKindCodec` 冻结逐 family base/token、reserved name、63-byte 截断与 typed-key suffix，artifact path 只取符号表结果 |
| 默认 profile 会因静态表情素材失败 | 默认改为 dual core，只要求不依赖新表情纹理的结构/骨骼动作；表情按 capability。strict avatar 为 opt-in，固定接受过门的 native/procedural 且 required-set 冲突时严格失败 |
| head/wrist/ankle 等 joint 来源不明 | A 阶段新增逐 joint observation 与 eligibility；wrist 允许系统性 unresolved |
| PSD bbox 可能被 alpha 裁小 | psd-tools 1.17.4 与项目 save_psd 透明边实测均保持 stored rectangle；保留精确断言 |
| RTMW dims、体积仍靠猜 | 下载官方包并记录真实 graph、大小、SHA-256；发现辅助 JSON 不一致 |
| RTMW 下载归错 owner | 改用 shared ONNX resolver + coherence-group inventory |
| Apache 代码许可被等同于模型权重许可 | 增加 Cocktail14 数据集许可证据 gate |
| 姿态模型评测只看关节是否在 mask 内 | 增加人工 GT、PCK/归一化误差/override 数 |
| SciPy/skeletonize 被误称为现有依赖 | 拆出明确的 auto-rig extras，使用成熟 skeletonize 实现 |
| “现有代码改动为零”不真实 | 列出 config/GUI/packaging/inventory/tests 触点并分阶段提交 |

---

## 设计冻结与实施边界

Revision 21 到此完成**规范层**设计：输入、公共数据模型、stage owner/resume、动作与表情语义、两种
exporter 的职责边界、正式 profile、失败语义、确定性规则和验收门都已有唯一选择。下面列出的事项是
需要用真实依赖、官方 runtime 或生产样本取得证据的 implementation/release gates，**不是留给实现者自由发挥的
开放设计项**。没有证据时必须按本文定义的错误停住，不能现场猜坐标、降低质量阈值、换格式或把 optional
能力冒充 required。

建议实施顺序固定为：

1. 先落地 schema、registry、JCS/manifest、startup validator 与最小 fixtures；
2. 独立完成 Spine golden probe 和 Live2D E0-core，签署 UV/frame/layout/runtime contracts；E0 不通过就停止
   对应 exporter，不用批量代码掩盖 writer 原语错误；
3. 实现 A/B 几何与确定性 mesh，再实现 C 的完整 Rig/ControlBinding/TexturePagePlan；
4. 实现 D/E 确定性物化与逐 item runtime gate，最后实现 G terminal finalizer 和 batch resume；
5. 用真实 see-through 批次冻结通过率、质量与容量阈值后，才允许正式双格式 release profile 写
   `completed`。

任何改变 frozen coordinate formula、role/control/primitive registry、internal/export ID codec、preset 曲线或 priority、stage ownership、
format version、native-variant eligibility/composite semantics、texture/mesh plan 或 validator threshold 的实现，都必须提高
相应 schema/plan/profile version并重跑其 golden/E0/production gates；不能只改文案或缓存 fingerprint 假装兼容。

## 实施前置验证与发布门

以下门项不妨碍按上面的顺序开始实现，但未关闭对应门前，不得宣称该 capability 或正式双格式交付完成：

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
- 在真实 NativeVariant 素材上按 `eye_closed/mouth_form/mouth_open`、768/1024/1280、主体有效 bbox 尺度
  和 cleaned base-support aspect ratio 分桶统计 `alpha_mass_ratio`、base alpha-mass、共同 feature scale、
  base-context radius、`intrusion_by_part`/face/other aggregates、coverage/intrusion rejection 的 p50/p95/max
  与人工真阳性/假阳性。v1 的 `k=1.5/4.0/12.0`，共同 `sqrt(mass_B)` estimator，eye
  `c=0.20,r=2..12 px`，mouth-form `c=1.00,r=4..24 px`，mouth-open `c=2.00,r=8..48 px` 在该 gate 前不可
  宣称已验证；尤其要单列高 aspect-ratio mouth，不能把 `r_min` 当作其充分性证明。若数据证伪，必须升
  `NativeVariantRoleEnvelope`/profile version并重跑 fixtures，不能在同名 registry 下自动调参。
- 单帧静态纹理生成 `blink/talk/happy/sad/surprised` procedural deform 的可接受率；不达标时
  `dual_runtime_avatar_v1` 应严格失败，而不是降低质量门槛。
- 可用于 release gate 的 Spine Editor 4.2 或匹配 runtime 环境及许可证安排。
- `Spine42UvAdapter` 的 atlas origin/V 轴/region offset 仍待官方 4.2 golden + runtime；
  `CubismV400UvAdapter` 已由 Revision 22 的 parser/Core/D3D11 五色 fixture 签署。共享 PNG 仍不回答
  Spine 数值 UV，D 阶段不得机械复用 Cubism 的 V 变换。
- Cubism Native D3D11 对 canonical straight-alpha PNG 的 loader/renderer 组合已用半透明 fixture 关闭；
  Web upload 路径仍待独立 release matrix，因为 Web sample 可能在 upload 时启用 premultiply。不能把
  Native 的结果或 Spine atlas `pma:false` 机械抄成 model3 字段。
- consistency/runtime 技术门已用官方 Core `06.00.0001` 精确 SHA-256 关闭，`04.02.0002` 继续只作
  低版本负例。开放项只剩发布主体的许可证判断以及未来新增 Core binary 的逐 SHA E0；普通安装不得
  捆绑 XKLive 或 SDK 的 Core DLL。
- MOC3 V4.00 的 root、axis-aligned rectangular warp (`quad_transforms=true`)、两层 rotation 与 ArtMesh
  parent frame 已在 `live2d-frames-v1` 关闭。一般曲面 warp、`quad_transforms=false`、warp-under-warp
  与 rotation-under-curved-warp 仍未验证，不能由 v1 结果外推。
- RotationDeformer 的 `angles/origin/scales/opacities/reflect` 初始化、parent indices、同节点
  angle+origin 九点插值以及两个不同 pivot 的 `rotation-stack-v1` 已通过官方 runtime。反射继续不支持，
  新增不同 `base_angle` 编码策略必须扩 E0 后升 schema，不能只看 consistency。
- 非中点 parameter default 的插值语义已由正例显式 rest key和负例缺 key签署；compiler 仍须逐 item
  判断两端插值能否精确还原 rest，不能无条件省略或无条件增加第三个 key。
- `Live2DFrameAttestation v1`、四个 semantic kernels、两份 MOC layout descriptor、纯函数向量、E0
  evidence 和 validator protocol 已物化为 packaged JCS attestation。普通 worker 只能消费和验证，不能
  临时重签；扩 Core allowlist 或修改 frame/UV/layout kernel 必须重跑 generator，production registry rows
  仍排除在 coordinate digest 外。
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
- [RFC 8785 JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)
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
- [Live2D Cubism Core API Reference（drawable UV/vertex runtime API）](https://docs.live2d.com/en/cubism-sdk-manual/cubism-core-api-reference/)
- [Live2D Cubism 4.2 Core change history（consistency API 与 04.02.0004 fixes）](https://docs.live2d.com/4.2/en/cubism-sdk-manual/core-updates/)
- [Live2D MOC3 consistency verification](https://docs.live2d.com/en/cubism-sdk-manual/moc3-consistency/)
- [Live2D Cubism Core distribution/license boundary](https://docs.live2d.com/en/cubism-sdk-manual/cubism-core/)
- [Live2D CubismWebSamples 4-r.4 official V4.00 goldens](https://github.com/Live2D/CubismWebSamples/tree/4-r.4/Samples/Resources)
- [Live2D Cubism 4.2 drawable vertex position / OpenGL coordinate direction](https://docs.live2d.com/4.2/en/cubism-sdk-manual/drawablevertexpositions/)
- [Live2D Cubism 4.2 draw order（0–1000，较大值在前）](https://docs.live2d.com/4.2/en/cubism-editor-manual/draworder/)
- [Live2D Cubism 4.2 compatibility](https://docs.live2d.com/4.2/en/cubism-sdk-manual/compatibility-with-cubism-4-2/)
- [Live2D texture atlas editor](https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/)
- [Live2D 4.2 texture alpha troubleshooting](https://docs.live2d.com/4.2/en/cubism-sdk-manual/texture-trouble-shooting/)
- [Live2D Web texture premultiply requirement](https://docs.live2d.com/en/cubism-sdk-manual/point-to-note/)
- [Cubism Web sample 动态加载 `getTextureCount()`](https://github.com/Live2D/CubismWebSamples/blob/b1de66b0b1f1cb881d95fb6158622aeb6a2827bd/Samples/TypeScript/Demo/src/lappmodel.ts#L532-L565)
- [Live2D Cubism External Application Integration](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration/)
