# Corridor Intelligence v1：图像候选区域证据评价

本阶段是独立 CPU 离线 Shadow。输入是已保存的 Spatial Grid v03，而不是重新运行 YOLO、SegFormer 或 Depth Anything V2。输出解释图像候选带之间的证据差异；没有导航命令、世界坐标路径、安全概率或碰撞预测。Risk Engine、RiskEvent、AlertEvent、HRTF、Live Camera 和 Hybrid/Laya 的正式行为均不接入本模块。

## 上游参考与独立实现

固定研究 [floatinglifeai/roana，7da28b7dd127f002986711ea62249b73de723739](https://github.com/floatinglifeai/roana/tree/7da28b7dd127f002986711ea62249b73de723739)，不以浮动分支的状态作结论。

- [CorridorGridFusion.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorGridFusion.kt)：参考先统一图像空间，再评价的结构。其逐帧 min/max 深度、检测框覆盖写固定强度不能迁移成米制障碍或概率。本站复用已有分位数网格与对象引用，不移植其数值融合规则。
- [CorridorPlanner.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorPlanner.kt)：实际固定 15×15，从底部中央向上一行的相邻三列递归，比较路径长度、横向净空及直行偏好；底部近深度与路径不足可 STOP。参考小规模候选和可解释比较，本站独立生成图像带，未移植递归 planner、深度门限或 LEFT/RIGHT/STRAIGHT/STOP。
- [CorridorStateMachine.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorStateMachine.kt)：核实非 STOP 连续确认、STOP 立即切换及 pending 重置。本阶段不实现该状态机，不以帧计数平滑正式提醒。
- [MotionQuality.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/MotionQuality.kt)：缺少 motion sample 时返回 STABLE/motion_unavailable 的宽松假设未沿用；本站缺少实测姿态时明确 UNKNOWN，默认 D 弃权。
- [ARCHITECTURE.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/ARCHITECTURE.md) 与 [STATUS.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/STATUS.md)：参考平台适配器与领域边界；上游设备/thermal gate 是上游记录，本站没有复现，更不代表道路可靠性。

第三方为 AGPL 项目。本阶段没有复制、改写或翻译其源码、测试、parity fixture，也未复制上游模型或媒体。所有新增 Python 算法及测试独立编写；依赖版本没有变更。更完整的先前架构审计见 [Roana 对照](roana_architecture_comparison.md)。

## 模块与接口

- `scene/corridor/schema.py`：不可变、类型明确的 `CandidateCorridor`、`CorridorEvidence`、`CandidateAssessment`、`ObjectCorridorRelation`、`CorridorAssessment`；结果 `sightguard.corridor-assessment.v1`，配置 `sightguard.corridor-config.v1`。
- `candidates.py`：确定性候选几何。`artifacts.py`：严格恢复现有 `SpatialGridFrame`，不另造替代网格协议。
- `engine.py`：`CorridorEvaluator.aggregate/evaluate_candidate/object_relations/assess`，可独立读取保存的同帧证据；不依赖风险、决策、UI、音频或模型对象。
- `pipeline/corridor.py`：预检、CPU 批量 A/B/C/D、逐阶段计时、JSONL、源与输出哈希及失败记录。
- `visualization/corridor.py` 与 HTML 模板：原始 RGB、阶梯候选边界、选中单元、对象引用、分项成本及未知原因。
- `scripts/run_corridor.py`、`scripts/validate_corridor.py`：复现与独立基线回归；默认单元测试不需要模型、下载或 GPU。

## Candidate Corridor 的精确定义

候选只表示原图像素中的观察区域，不表示可行走通道。默认 `IMAGE_LEFT/CENTER/RIGHT` 三个模板；有三个模板不等于有三个可评价走廊。

对于 R×C 网格：首行为 `floor(0.4R)`，观察到最后一行；下部关注区从 `floor(0.8R)` 开始。默认 15×15 时是第 6–14 行，第 12–14 行额外作为下部证据。每行插值位置 t=(r−r0)/(R−1−r0)：中心列在顶端比例 (0.2,0.5,0.8) 与底端 (0.35,0.5,0.65) 之间插值，再乘 C−1；宽度在 0.2C 与 C/3 之间插值并取 Python 确定性 round，至少一列，边界截断到图像内。

候选包括 `cell_ids`、每行半开矩形、其精确阶梯外边界 `polygon_xy` 和 `near_cell_ids`。边界面积等于所选单元总面积，非整除图像尺寸仍沿源网格边界。默认各带 36 单元、下部各 14 单元；底部带可重叠。A/B/C/D 共享完全相同的几何与 FrameKey。几何生成不使用语义、深度或目标来暗中预选候选。

所有候选依赖“下部图像值得更多注意”的工程假设，不是米制近场。没有相机标定、姿态、足迹宽度或视角补偿；既不能推断用户当前位置，也不能把二维连通性宣称为现实可通过。Spatial Grid v1 没有实测姿态字段；即使删除限制字符串，也不会把姿态当成已知。

## Ground / Obstacle / Depth / UNKNOWN

每个单元使用像素面积和行关注权重 `w_i=area_i × [1+gain×(r_i−r0)/(R−1−r0)]`，gain 默认 1。下文均值是所选单元上的加权均值；下部均值仅使用下部单元。

1. **Ground**：保留 raw support、floor=3、road=6、sidewalk=11 的分别占比。类别支持因子为 floor 0.8、road 0.5、sidewalk 1.0；每格在三类占比内求加权因子，再乘其原支持度，得到 adjusted support。没有三类占比则为 0。road 支持更弱是显式工程参数，不是道路能否安全行走的事实。
2. **Continuity**：从候选底行中 supported 且具有原 bottom-connected component ID 的格开始，向上一行仅延续相邻列（差≤1）且共享组件 ID 的格。连续支持行数/候选总行数作为连续度；断开的所有上方行保留在 interrupted_rows。它只描述图像组件连续性，既不输出路线也不证明真实净空。
3. **Obstacle**：每格取达到原检测门限的框覆盖比例的最大值，聚合出候选覆盖与下部覆盖。这是最大框覆盖代理，不是多框 union、instance mask、障碍概率或真实足迹。目标类别、confidence 和 session 内 track ID 逐个保存，重叠目标不重复累加成超过 1 的占用。
4. **Depth**：复用同帧相对逆深度的 frame percentile，值大仅表示帧内相对更近。下部项是 `mean(bbox_proxy_i × percentile_i)`，变化项为 `mean(min(1, spread_fraction_i))`。深度覆盖率按每格有效像素数/格面积聚合，不能因只有一个有效像素而记整格充分。缺源、退化帧量程、缺相对统计返回 null，不用 0 假装有深度；近项只在有效下部单元上计算，并保留整体缺测比例。
5. **UNKNOWN**：每格取地面未知、检测源不可用及所需深度缺测比例的最大值，再加权聚合。被消融排除的深度不构成 B 的未知。姿态 UNKNOWN 单独保存及门禁，未与 cell UNKNOWN 混为一个概率。
6. **Conflict**：B 只使用 ground/bbox candidate overlap；C/D 加入原 relative-depth-variation reason。没有把深度变化解释为物体边界、坡度或碰撞。B 不偷用深度冲突、数值或对象深度。

原始统计、有效覆盖、来源状态、quality reasons 与分项都保存，质量不足不会删除候选几何或隐藏对象。

## Eligibility 与 Cost 分离

A 只有几何，没有可评价性证据，全部 `INELIGIBLE`。B/C 的基础门禁：semantic/detection 源有效、ground support≥0.2、底行 ground≥0.15、连续度≥0.2。C 是允许保留缺深度分项的研究消融，缺项显式 null，不重归一化，不可理解为“未知是空闲”。

D 在基础门禁上增加 unknown≤0.45、conflict≤0.4、下部框覆盖≤0.65、semantic confidence≥0.45、有效深度像素覆盖≥0.8、实测姿态要求。当前协议没有实测姿态，默认 `require_observed_orientation: true` 导致本次 D 全部 ABSTAIN。这是缺少证据的实际结果，不能称为已取得方向可靠性。关闭姿态门禁仅作为显式研究配置；结果仍标记 UNKNOWN，本次八视频主实验没有关闭门禁或调参。

成本分项：

```text
ground     = 1 − adjusted_ground_support
continuity = 1 − ground_continuity
obstacle   = bbox_proxy_coverage
depth      = 0.75 × near_relative_depth_proxy + 0.25 × relative_depth_variation
unknown    = unknown_exposure
conflict   = conflict_coverage
cost       = .25 ground + .20 continuity + .20 obstacle
           + .15 depth  + .15 unknown    + .05 conflict
```

这些权重只用于工程可解释消融，未训练、未标定、未通过通行 GT 验证。配置校验所有比例有限且在范围内，六项权重和为 1。A 分项全部 null；B depth 为 null。被门禁拒绝的候选仍保存可用分项及加权贡献，但 `total_cost=null`，不进入排名。

可评价候选按成本排序；最低成本 0.03 内存在多个候选则 `TIED`，best=null，保留 tied_candidates。无可评价候选则 `ABSTAIN`，保留各候选及帧级原因。只有唯一支持候选才输出研究字段 `best_supported_candidate`；它不是 LEFT/RIGHT 指令。并列时稳定字典排序只为序列化，不意味着选择一条方向。

## Object–Corridor Relation 与未来接口

用原 `ObjectGridRelation.cell_coverage` 与候选 cell_ids 相交，保存原 detection index、nullable track ID、class/confidence、相交单元、候选内原裁切框比例、候选/下部像素面积框代理、对象原 ground fraction、同帧对象深度 percentile。邻近且未落在该框及其他检测覆盖中的单元提供背景 percentile 均值，存在时保存对象减背景的相对对比；不存在则 null 和 background_depth_unavailable。

框内包含背景、分割不提供 instance mask，因此关系始终 limited。上述比例与背景对比不是目标运动、接地位置、世界距离或碰撞时间。没有聚合跨帧历史，也没有把跟踪 ID 当作稳定真实身份。

`event_reference` 默认 null：没有可靠的一对一同帧 RiskEvent 索引时不猜测关联，不创建新的正式事件。未来 Dynamic Decision 应通过独立适配层读取相同 FrameKey 的这些可序列化记录，显式验证事件对应、时间来源和不确定性，再做独立研究；本阶段未实现该适配层、状态机或提醒政策。

## 保存、验证与边界

运行前完整核验八视频所有 frame、时间戳、源尺寸、225 格完整性、对象双向引用、独立 object_relations 文件一致性，以及父代码、manifest、tracks/semantic/depth JSONL、视频、语义/深度 NPZ、父模型运行 metadata 的 SHA256。输入原样保留；新目录必须不存在，错误写入 run_metadata，不覆盖旧结果。

输出包括各视频 corridor_frames/candidate_corridors/object_relations JSONL、summary、RGB/标注预览、可独立 HTTP 打开的 viewer；根目录包含 config/source/runtime/input/output 哈希、timings、ablation_summary、failure_cases 及独立 validation。时间逻辑使用视频 PTS，性能计时仅用于工程诊断。

已知缺口：相机标定/实测姿态、尺度与用户足迹、instance 地面接触、跨帧尺度及候选身份、动态运动风险、独立通行与风险 GT、伴行监督和真实道路验证。现在的 2 Hz 相邻候选身份变化只是图像输出诊断，不能称路线稳定或行为改善。实验与复现详见 [验证及阶段报告](../archive/corridor_intelligence_validation.md)。
