# Spatial Grid Fusion v1

Spatial Grid 是**图像空间证据表示，不是米制三维占用地图**。本模块离线
Shadow 运行，不调用 Risk Engine、Event Manager、AlertScheduler 或音频适配器，
不输出导航、安全通行、真实接地、碰撞概率或 TTC。

## 来源调查与实现边界

参考仓库为 [floatinglifeai/roana](https://github.com/floatinglifeai/roana/tree/7da28b7dd127f002986711ea62249b73de723739)，
固定 revision `7da28b7dd127f002986711ea62249b73de723739`。

- [CorridorGridFusion.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorGridFusion.kt)：
  框按归一化坐标投影到网格，高置信度框覆盖的格被覆盖为障碍代理值。
- [CorridorPlanner.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorPlanner.kt)：
  DepthGrid 每帧 min/max 归一化后按格取平均；规划搜索上一行相邻三格，
  比较长度、横向净空和直行偏好，并由端点偏移产生方向命令。本轮不实现规划。
- [CorridorContract.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorContract.kt)：
  15×15 是固定工程契约。在所读代码和文档中没有找到该尺寸优于其他尺寸的实验证据。
- [MotionQuality.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/MotionQuality.kt)：
  缺少 motion sample 时返回 STABLE/motion_unavailable；本项目保持相机姿态未知。
- [ARCHITECTURE.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/ARCHITECTURE.md) 与
  [STATUS.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/STATUS.md)：
  原项目还有平台推理、走廊状态机和反馈链路；其平台验证状态不能转移为 SightGuard 的验证结果。

Roana 的网格没有逐单元 UNKNOWN、时间身份与来源哈希；常量深度映射为零。
框全域覆盖要求框与深度属于相同图像坐标，但不能证明实际物体足迹或可通行性。
这些局限是本轮独立证据契约的设计依据。

[LICENSE](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/LICENSE) 和
[NOTICE.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/NOTICE.md)
声明 AGPL-3.0-or-later。本轮没有复制、翻译或移植 Roana 实现、测试、fixture。
仅借鉴共同图像坐标网格与检测覆盖的数学思想，使用本项目契约、NumPy 和 OpenCV 独立实现。
没有采用其障碍代理常数、规划阈值或方向状态机。

## 模块与数据契约

- `scene/spatial_grid/schema.py`：版本化配置、FrameKey、EvidenceSource、各类证据与结果。
- `scene/spatial_grid/engine.py`：无状态 NumPy 聚合，不依赖模型、UI、风险或音频。
- `scene/spatial_grid/artifacts.py`：哈希验证、严格身份校验、掩码对齐、DepthFrameResult 适配。
- `pipeline/spatial_semantics.py`：仅在缺少原始掩码时，复用现有 SegFormer 适配器生成新缓存。
- `pipeline/spatial_grid.py`：保存数据离线融合、消融、计时、资源与失败记录。
- `visualization/spatial_grid.py`：六面板真实对照和按需加载的离线 HTML 单元/对象检查器。

网格结果 schema 为 `sightguard.spatial-grid.v1`，配置为
`sightguard.spatial-grid-config.v1`，新语义缓存为 `sightguard.spatial-semantic.v1`。
原始 tracks `1.0`、Path Relevance v1、Depth v1 数据格式不变。

`FrameKey(video_id, frame_index, timestamp_ms, width, height)` 必须完全一致。
时间为原视频 PTS，不能使用推理墙钟或沿用上一帧。Depth 消费先调用
`DepthFrameResult.matches_frame`，再调用现有 `load_depth_map` 验证 SHA、float32、
尺寸和有限值。存储读取发现非有限深度会拒绝整张矩阵；纯算法接口还可处理局部非有限值。

各来源保留独立 status、quality、schema、文件引用、SHA 和原因。
原生 288×512 的 SegFormer class/confidence/component 数组无损保存；三个数组均用
nearest 对齐到源图，防止在类别边界跨类别混合置信度。此方法仍有原生分辨率的量化误差。
原有 JPG、polygon、简化 scene JSON 不能恢复原始掩码，因而不能替代模型数组。

## 网格坐标

默认 15 行×15 列，可通过 `configs/spatial_grid.yaml` 修改。
坐标左上角为原点，x 向右、y 向下。格子为半开区间：

```text
x1 = floor(col * width / cols)
x2 = floor((col + 1) * width / cols)
y1 = floor(row * height / rows)
y2 = floor((row + 1) * height / rows)
cell_id = row * cols + col
```

非整除尺寸覆盖每个像素一次，不重复最后一列/行。图片必须至少每格一个像素。
网格尺寸不能解释为现实空间长度；不同距离位置的单元对应不同的物理面积。

## 单元证据与状态

每格保存 bounds、row/col/cell_id、state、quality，以及以下独立证据：

1. Ground：高于配置置信度的 floor=3、road=6、sidewalk=11 像素支持比例；
   原始 argmax 类别比例；支持像素平均置信度；候选组件比例、当前帧组件 ID 和 scene quality。
   类别比例包括低置信度 argmax，不能与受阈值筛选的支持比例混淆。
2. Depth：有效像素数量、负值/非有限/无效比例、原值 median/q10/q90、帧内百分位与 spread。
3. Detection：所有相交目标的 detection_index、会话 track_id、class、confidence，
   bbox 对格和格对 clipped bbox 的覆盖比例，候选标志与 `limited_bbox_proxy` 质量。
4. 独立 source_status、冲突与 UNKNOWN 原因、每个来源的路径和 SHA。

状态使用简单且可解释的证据门限，不做安全评分：

- `GROUND_EVIDENCE`：有效地面支持比例 ≥ 配置门限。
- `OCCUPANCY_EVIDENCE`：至少一个有效检测达到置信度和 cell 覆盖门限。
- `MIXED`：两类证据均达到门限，明确记录 ground_and_bbox_candidate_overlap。
- `UNKNOWN`：未达到上述证据条件。Depth Only 即使数值有效，仍无地面/占用标签。

深度变化只添加 `relative_depth_variation_not_object_boundary` 原因，不能把格子升级为障碍。
任何状态都不能解释为 FREE、HIGH 风险或安全概率。多个框不相加为占用概率。

语义 UNKNOWN 拒绝使用其地面数组，但可保留同帧有效深度/检测；深度 UNKNOWN 保留语义。
低 scene quality、局部无效置信度、负深度、错误检测和缺失来源均显式进入 UNKNOWN/limited 原因。
GROUND_EVIDENCE 格也可能含未知或受限证据，不能只读取 state 而忽略 quality 和 reasons。
帧总 status 保守为 limited 或 unknown，没有全局 valid 覆盖局部缺失。

候选连通域复用本项目 `estimate_corridors`，不另建走廊规划器：
高置信度地面、3×3 close、8 连通、面积和垂直延伸门限、接近画面底部区域。
component IDs 是新缓存中逐帧的局部连通域标签，不是跨帧身份；形态学连接像素与原始语义支持分开保存。
原来估计器的 quality/reasons 原样保留；其旧字段名 corridor 不代表本轮具有可通行路线。

## 相对深度

较大原值表示相对更近；所有数值均非度量。有效统计像素为 finite 且 ≥0。
负值按双三次重采样边界无效值排除，不修改原矩阵、不补零、不解释为负距离。
median/q10/q90 使用 NumPy 稳健统计。帧内有效数组排序，median 百分位使用左右秩平均。
常量图的百分位与 spread 为 null，并标记 limited，避免凭空生成近远结构。
spread 定义为单元 `(q90-q10)` / 同帧全图 `(q90-q10)`，门限为工程诊断值。
排序副本一次转换为 float64，避免每个标量查询隐式复制整张 float32 图；原始数组不变。

显示图使用帧内 p01-p99 色阶，与原值、排序百分位和单元统计分开。
不比较不同帧独立归一化的深度，不推算速度、米制距离、接触、净空或 TTC。
玻璃、反射、遮挡、细杆和低矮台阶的错误不会被有效数值自动排除。

## Object–Grid Relation

每个有效检测输出 clipped bbox、相交格分布、原 bbox 与 cell/bbox 覆盖比例、
框内相对深度摘要、地面支持比例、组件 ID 和 image_left/image_center/image_right。
框统计按像素中心落入连续框的规则取样；极细框可有几何覆盖但没有统计像素。
方向仅为图像位置三分区，不是转向建议。
缺 track_id 保留 null；错误 confidence/class/geometry 生成 unknown 关系与原因，不借用别的目标。

检测框含背景、椅腿间空隙和被遮挡地面。框深度中位数也可能代表背景，
不是目标表面距离；ground_fraction 不证明对象接地。保留 MIXED 以供核查，
不将 bbox 填为实心、不创建 event_id、不修改 BoT-SORT 身份或事件生命周期。
`detection_reference` + 帧身份 + detection_index 提供现有事件体系以后可读取的溯源入口。

## 输出、性能与下一阶段接口

新输出目录必须不存在。每视频保存 `spatial_frames.jsonl`（完整 cells）、
`object_relations.jsonl`、`ablation_metrics.jsonl`、metadata、summary、timings、previews、
逐帧 inspector JS；批次保存 summary 和 index.html。原始数组只保存一次，单元仅保存引用与摘要。
未完成批次记录 failed/error；来源 UNKNOWN 与模型执行异常分别记录，不能伪装为完整观测。
同源 A/B/C/D 固定帧身份；被排除来源使用 excluded，区别于读取失败。

复杂度主要为全帧排序 O(P log P)、单元统计 O(P)、框覆盖 O(DG)、框内统计 O(Σ bbox pixels)。
多个框重复覆盖的统计成本和保存/可视化成本都必须计入性能。当前实现优先可解释、可重放。
实际数据、耗时、资源、测试与复现见 [spatial_grid_validation.md](../archive/spatial_grid_validation.md)。

未来候选走廊评价可以直接读取 FrameKey、cells 的半开坐标、三类证据与局部 quality/reasons、
组件联系、ObjectGridRelation 和源文件哈希，并复用保存输入作对照。
在规划前仍需独立空间标注、姿态/标定、可见性与遮挡处理，以及时间有效性契约。
本轮没有实现 Corridor Planner、动态引导、音频调度或移动端应用。
