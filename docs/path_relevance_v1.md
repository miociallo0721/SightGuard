# SightGuard Path Relevance v1

状态：离线研究型 Shadow 分析。Path Relevance 是目标与模型估计语义地面区域的图像空间关系，不是 Risk Score、碰撞概率、已验证的通行路线或安全保证。它补充盲杖、导盲犬和定向行走训练，不替代它们。

## 问题与范围

现有 Risk Engine 使用检测、轨迹、位置、尺度和运动等特征产生风险事件及提醒，但没有语义地面区域这一类空间证据。本阶段复用已保存的 YOLO26n / BoT-SORT tracks、Risk Engine 帧结果、事件和提醒，只对原视频的定时采样帧运行语义分割。没有重新运行 YOLO，也没有修改风险阈值、Alert Policy、音频、实时摄像头路径或默认事件关联策略。

本阶段只回答一个较窄的问题：**检测框在多大程度上覆盖了当前帧里模型估计的、从图像下方延伸出来的地面语义区域？** 这不回答真实地面投影、行人意图、可通行宽度、障碍物深度或碰撞时间。

## 开源算法研究、来源和许可证

- 主要空间结构参考是 [Roana](https://github.com/floatinglifeai/roana) 在提交 [`7da28b7dd127f002986711ea62249b73de723739`](https://github.com/floatinglifeai/roana/tree/7da28b7dd127f002986711ea62249b73de723739) 的实际代码。`CorridorGridFusion.kt` 将深度图缩至 15×15 网格，并把检测框覆盖单元标成高占用值；`CorridorPlanner.kt` 从网格底部中央搜索候选单元路径，以近障、长度和方向规则选 STOP / LEFT / STRAIGHT / RIGHT。它是走廊选择回路，不是 SightGuard 的 track/event 生命周期算法。项目许可为 AGPL-3.0-or-later。
- 本次**没有复制或移植 Roana 源码**，也没有直接集成 Roana 的深度网格或导航策略。只借鉴了“显式建模候选通行空间，再度量目标与空间的关系”这一研究思路；SightGuard 的场景估计和关联均为本仓库独立实现。
- 实际参与推理的模型为 OpenMMLab MMSegmentation model zoo 的 SegFormer MIT-B0、ADE20K、512×512、160k checkpoint。官方 [SegFormer 模型表](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/README.md) 列出该配置并链接权重；[MMSegmentation 仓库](https://github.com/open-mmlab/mmsegmentation)标注 Apache-2.0。仓库代码许可证不自动证明单独下载的 checkpoint 或训练数据具有相同许可；模型表未单独注明该 checkpoint 的许可，本项目记录为“未独立核实”，不据此声称权重可再分发。
- 权重文件：`models/path_relevance/segformer_mit_b0_ade20k_160k.pth`，15,072,692 bytes，SHA-256 `8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`。源链接及哈希也写入每次输出的 `run_metadata.json`。权重在 `.gitignore` 中，不提交 Git。

因此，对“是否融合开源代码”的准确回答是：**真实 OpenMMLab 发布的 SegFormer 权重实际参与了场景推理；没有把 Roana 或 MMSegmentation 源代码复制进 SightGuard。** 运行时通过 Transformers 加载 SegFormer MIT-B0 encoder，并由项目内的 checkpoint-compatible adapter 对齐 OpenMMLab decode head 权重。

## 场景感知与候选区域

输入帧保持宽高比，将宽度缩放至 512，高度取 32 的倍数；BGR 转 RGB 后按常用 ImageNet 均值和标准差归一化。模型生成 150 类 ADE20K 的每像素 softmax 概率，保存 argmax 类别图和最大类别概率图。代码路径：`src/sightguard/path_relevance/model.py`。

候选地面语义由 ADE20K 类别 `floor=3`、`road=6`、`sidewalk=11` 的像素组成。仅保留置信度不低于 0.35 的像素，进行 3×3 闭运算，再用 8 邻接连通域筛选：

1. 连通域面积至少占图像的 0.5%。
2. 连通域在图像下部 18% 范围内出现。
3. 连通域纵向延伸至少占图像高度 35%。

所有符合条件的分离区域均保留，不选固定中心 ROI，不假设唯一通道或中心线。每个分量输出 mask、外接框、多边形、面积和垂直延伸。场景置信度是保留地面像素平均类别置信度乘以 `min(最大垂直延伸 / 0.7, 1)`。分数至少 0.65 标为 `high`，否则为 `limited`。没有合格分量、模型不可用、推理失败、结果无效或时间戳不满足约束时返回 `unknown`，而不是“无风险”。

这是 ADE20K 语义标签筛选，不是地面可行驶性分类。语义上的 road / sidewalk / floor 可能被车辆、家具或遮挡覆盖；本版本没有相机标定、姿态补偿、度量深度、显式光流相机运动检测，也没有可靠的动态遮挡分解。快速转头或遮挡不一定会被单独识别，需人工复核。

## 检测框关联和 Path Relevance

每个采样帧按 `frame_index` 连接保存的 tracks 与 Risk Engine `risk_frames.jsonl`；两边视频时间戳必须相差不超过 1 ms。event_id 取同帧相同 `detection_index` 的 Risk observation。采样间隔为 0.5 秒，结果 TTL 为 500 ms；这里对每个采样帧重新分割原帧，没有将上一采样帧的分割结果复用成当前证据。track_id 仍只是单段视频会话内的估计身份。

将检测框缩放到分割 mask 尺寸，定义：

```text
intersection    = 检测框中候选地面像素比例
contact_support = 检测框下部区域中候选地面像素比例
score           = clip(0.70 × intersection + 0.30 × contact_support, 0, 1)
```

行人使用检测框下部 28% 作为 `contact_support` 区，其他类别使用下部 35%。等级阈值来自 `configs/path_relevance.yaml`：`LOW < 0.20`、`MEDIUM >= 0.20`、`HIGH >= 0.55`。场景为 `limited` 且得分低于 0.20 时输出 `UNKNOWN` 并将分数置空，避免低质量画面把低重叠解释成“与通道无关”。非有限、退化或框外检测框也为 `UNKNOWN`。

类别只影响接地点启发式及置信度折减，不改变上述分数权重：person 的接地点先验 0.60；chair / bench / couch / bed / dining table / toilet 为 0.45；car / truck / bus / motorcycle / bicycle 为 0.35；其他类别为 0.20。关联置信度为 `scene_confidence × (0.55 + 0.45 × contact_prior) × detection_confidence`，再截断至 0..1。缺失检测置信度时不额外折减。以上先验是未经标定的工程启发式，不代表类别风险等级。

`estimated_spatial_relation` 另按 mask-overlap 输出 `corridor_overlap`（至少 0.15）、`partial_corridor_overlap`（大于 0.01）、`adjacent_to_corridor` 或 `outside_corridor`。它不推断米制距离，也不表示目标真的阻挡了人的运动。对象结果含 video/timestamp/frame/track/event、框、类别、交叠、等级、置信度、不确定性、原因、证据来源和推理耗时，写入版本化 JSONL。

## Baseline、对照与 Shadow 边界

离线对照使用相同的已保存 tracks 和 Risk observation：

- **A Baseline**：原有 Risk Score 与 AlertEvent，只读取，不重算新提醒。
- **B Geometry Baseline**：检测框与静态中心下部 ROI（画面水平中间三分之一、垂直下部 45%）的交叠，只作几何对照，不冒充 Path Relevance。
- **C Path Relevance**：真实 SegFormer ADE20K 结果和框区域关系。
- **D Shadow**：记录 `baseline_risk_score × path_relevance_score` 作为候选优先级（两者缺一时为 null），并生成研究建议。

Baseline HIGH 总是 `preserve_baseline_high`；Path Relevance 未知时 `retain_baseline_path_unknown`；高相关候选为 `retain_path_related_candidate`；其余通常 `manual_review`。仅低相关且关联置信度至少 0.55 的非 HIGH 项会给出 `lower_shadow_priority_research_only`。这些建议按采样目标观察计数，不是 AlertEvent 个数，不会自动抑制提醒。输出的 `production_decision_changed` 始终为 false；原始 `alerts.jsonl` 不被改写，默认实时路径没有启用模型。

## 八段视频观察

量化指标、每视频摘要和复现命令见 [path_relevance_validation.md](path_relevance_validation.md)。最终 RTX 2050 批次 v04 共有 1,123 个目标观察：125 HIGH、376 MEDIUM、240 LOW、382 UNKNOWN；741 个观察有定义分数，按观察加权平均为 0.319。Path Relevance 阈值只是本阶段实验分层，不是风险阈值，也没有 Ground Truth 支撑其准确度。

在人工浏览 7.5 秒关键帧时，V05 的模型地面区域较宽，连入道路和人行道大片区域，无法形成窄的个体通行走廊；V03 中车辆框、摩托车框与模型地面区域的重叠说明粗略 bbox 交叠可能受遮挡和框高度影响；V04 只有 3/30 场景帧质量为 high，108 个对象观察中有 100 个为 UNKNOWN，显示保守弃权也会降低输出覆盖。V08 的地面叠加视觉上较连贯，但仍没有独立标签可验证。以上是定性复核，不是完整错误率统计；没有足够证据把任何样本断言为真实误报、漏报或安全改善。

## Research Console

新增只读“通行相关性 · Shadow”标签页，读取 `outputs/path_relevance/<run>/` 的场景、对象和 Shadow JSONL，支持选择运行、视频、采样帧、track 和 event，查看叠加图、Baseline Risk、Path Relevance、原因及模型耗时。它与 Annotation Studio 的盲标界面隔离；没有把模型建议回填成 Ground Truth。

## 限制与下一步

- UrbanEgo 八段片段共 120 秒、2,400 帧，均来自同一次原始拍摄并属于 development；240 个 2 Hz 样本不是 240 个独立场景。
- ADE20K 类别置信度不是走路可行性置信度。人行道或道路类别不能证明地面无遮挡或安全；静止摄像头/机器人类走廊算法与移动头戴视角输入条件不同。
- bbox 交叠不等于接地点、空间占用或行人意图。检测框含大量背景，目标姿态和遮挡会改变重叠比例；track_id 也不构成实体身份证明。
- 未估计相机内参、姿态、真实深度或地面平面；无法做透视校正、跨帧运动补偿、距离和碰撞时间估计。
- 尚无经过独立复核的视觉属性标签。`datasets/real_world/annotations/visual_tasks.json` 仍是 `pending_human_annotation` 且标签为 null。因此不报告 Accuracy、Precision、Recall、F1、误报下降或安全收益。
- 后续应先对候选地面分割和目标关系建立独立盲标，再评价画面遮挡、相机运动和类别分层；若扩展到相机几何、深度或实时使用，应作为单独阶段并重新验证。

这套实验提供了原 Risk Engine 未显式表示的语义地面交叠特征，但当前只能作为可复现研究信息显示，不能指导真实行走或改变正式提醒。
