> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard Algorithm Refinement v1 — 详细任务报告

**项目：** SightGuard（视界）  
**报告日期：** 2026-10-11  
**代码分支：** `codex/annotation-studio`  
**实现提交：** `bf75e7386e0c3b4b57b29b4278ea0a37795c3ba4`  
**研究范围：** 风险事件关联与提醒行为；不评价真实道路安全能力。

## 1. 执行摘要

本任务检查了现有事件关联、提醒历史、风险处理和八段 UrbanEgo 感知日志，研究 CaBot People 与 Roana 的相关实现，针对类别预测短时变化导致的事件分裂，编写回归测试并实现独立的 Event Association v2 实验开关，随后在相同的 2,400 帧感知轨迹上重放 Baseline 和实验策略。

确认的软件机制是：Baseline 的 EventManager 对 track_id 续接和几何关联都要求检测类别相同。类别发生变化时，即使 tracker ID 保持稳定，该观察也不能按 track_id 续接原事件。Video 03 的 track 1 在类别由 truck 变为 car 后触发了这一规则，产生新的 E000004。它说明软件行为可以制造事件分裂；但没有 Ground Truth 能证明分裂的事件属于同一个物体，所以没有将该机制描述为已确认的身份错误。

实验策略仅允许满足身份唯一性和空间一致性的同一 tracker ID 跨类别续接。最终版本增加 IoU 下限，拒绝初轮回放中发现的低重叠续接。默认配置关闭该策略，原提醒历史仍绑定 event_id。

实验组比 Baseline 少 39 个风险事件、少 2 条提醒、少 530 个跨事件 track 冲突观察。与此同时，HIGH 提醒总数从 120 降为 116，Video 03 的 E000004 首次 HIGH 提醒不再出现，另有多条 HIGH 提醒发生变化。由于无法判定相关 track 是否表示同一真实目标，实验结果只能说明程序行为改变，不能证明身份判断或提醒质量改善。Baseline 保持为默认策略。

## 2. 任务目标与边界

目标是分析相同 track ID 参与多个首次提醒的可能原因，并用一个小范围、可开关、可回放的策略验证类别抖动与事件分裂之间的关系。实现没有加入模型、重新运行目标检测、修改默认风险阈值、提醒冷却、音频输出或 HRTF 行为。

八段 UrbanEgo 视频都来自同一次原始拍摄，每段 15 秒，共 120 秒、2,400 帧。当前没有经人工复核的风险事件 Ground Truth；因此提醒条数、风险等级和事件片段数量不是误报率、召回率或安全性指标。

## 3. 开发前检查与代码路径

### 3.1 Git 和工作区

- 开发分支为 `codex/annotation-studio`；开发前 HEAD 为 `1324d97`。
- 开发前没有已跟踪文件的未提交修改。已有两个用户未跟踪文件：`docs/archive/hybrid_intelligence_delivery_report.md` 和 `docs/archive/stage_report_2026-10-11.md`。
- 没有编辑 `AGENTS.md`，没有覆盖、暂存或提交上述两份用户文档。
- 本次实现提交为 `bf75e7386e0c3b4b57b29b4278ea0a37795c3ba4`；详细报告是本轮新写的文档，尚未提交。

### 3.2 阅读和诊断的本地组件

检查了 `decision/events.py`、`decision/alerts.py`、风险配置和评分/引擎接口、感知数据适配、`pipeline/assess.py`、CLI、现有关联和提醒测试，以及 `scripts/diagnose_alert_behavior.py` 和 Alert Behavior Diagnosis 文档。

代码层面的要点：

1. EventManager 先尝试 tracker ID 匹配，再尝试同类别几何匹配。候选必须满足有效时间窗口及相应几何条件；双向唯一最佳匹配处理一对一约束。歧义端点会保持未匹配，之后可能建立新事件。
2. `track_id` 在视频会话内使用，表示追踪器输出身份，不是真实对象 ID。一个真实对象可能换 ID，也可能因 tracker 错误让不同对象暂时共用一个 ID。
3. LOST → ACTIVE 恢复会继续同一 event_id。提醒调度历史按 event_id 保存，不会因 LOST/ACTIVE 转换而清空；新 event_id 会形成独立提醒历史。
4. 既有诊断中的 `same-frame competition=0` 不代表没有身份关联问题：同一 track ID 可在不同事件生命周期中出现，也可先后出现在不同事件的首次提醒中。

## 4. 开源项目调查

完整源码路径、版本和许可记录见 [open_source_algorithm_review.md](../research/open_source_algorithm_review.md)。本次未克隆大型仓库，也没有复制第三方代码。

### CaBot People

检查了仓库提交 `4de04120e6d8f833b41e10c514938b80cc0f5d7a` 的 `tracker_sort_3d.py` 和 `track_abstract_people.py`。项目自己的跟踪器围绕 3D 鸟瞰中心、Kalman 预测圆重叠度、阈值过滤与一对一 Hungarian assignment 建立匹配；丢失轨迹有保留及过期机制。多相机抽象人物关联使用 3D 空间邻近信息。线性分配由 SciPy 提供，Kalman Filter 使用 FilterPy。

可借鉴的是明确的冲突拒配、一对一匹配和有界的丢失恢复。其 3D 空间依赖不适合直接移植到当前单目框接口，也不是 BoT-SORT 重识别实现。项目为 MIT 许可。

### Roana

检查了仓库提交 `7da28b7dd127f002986711ea62249b73de723739` 的 `CorridorGridFusion.kt`、`CorridorPlanner.kt` 和 `STATUS.md`。已实现部分把检测框融合进深度网格，再按候选走廊和路径代价给出方向或停止决策。实际处理的是深度网格与走廊选择，不是 track/event 身份维护、提醒历史继承或按无关目标去重。其单目深度图不是校准米制距离。项目为 AGPL-3.0-or-later；代码复用需单独审查许可义务。

## 5. Baseline 案例复现

### 5.1 Video 02：E000006 / E000020

Baseline 的关键状态序列：

1. `E000006` 于 0ms 以 track 9 建立，并在 600ms 产生 MEDIUM 首次提醒（score 约 0.488617）。该事件多次 LOST/ACTIVE 转换。
2. `E000020` 于 1,450ms 以 track 94 建立。
3. 1,550ms，E000006 通过几何关联从 track 9 切换到 track 6，并保留来源 `[9, 6]`。
4. 1,650ms，E000020 通过几何关联从 track 94 接入 track 9，并保留来源 `[94, 9]`。
5. E000020 于 1,850ms 产生第二条 MEDIUM 首次提醒（score 约 0.589694）。这时 E000006 与 E000020 的生命周期重叠；E000006 在 2,400ms 才解析结束。

此序列不是单纯“旧事件超时后新建”。它显示两个活动事件曾分别保有同一 track 的历史或当前关联，但逐帧的一对一分配没有允许同一 observation 同时进入两个事件。现有 JSON 无法判断 track 9、6、94 是同一行人、两个人之间的 ID 迁移，还是遮挡/相邻目标带来的错误几何关联。

最终 v2 在该案例的创建、关联、事件状态和提醒时刻均与 Baseline 相同：600ms 和 1,850ms 两次提醒仍在。V02 总风险事件仅从 123 减到 121，55 条提醒不变。该类别抖动策略没有解决此处主要的 tracker ID / 几何迁移问题。

### 5.2 Video 03：track 1 与多个事件

Baseline 的前段状态序列：

1. E000003 于 0ms 以 truck / track 1 建立，250ms 确认并产生 HIGH 首次提醒（score 0.743952）。
2. 350ms 检测变为 car，但仍为 track 1。类别硬门控拒绝 track ID 匹配，于是创建 E000004；原 E000003 进入 LOST。
3. 400ms 的 truck 观察通过几何关联恢复 E000003；之后车类观察又继续更新 E000004。E000004 于 850ms 产生 HIGH 首次提醒（score 0.8）。
4. 2,600ms 左右出现新一轮类别和关联竞争。Baseline 的候选 E000013 起始于 track 29 / motorcycle，后续 track 1 的 motorcycle 观察也经几何关联加入该事件。E000013 于 3,550ms 首次 HIGH 提醒，来源 track IDs 为 `[29, 1]`，score 0.722081。

v2 在 350ms 通过 `track_id_class_switch` 把 car 观察续接到 E000003，E000004 不再建立。由于 E000003 已在 250ms 提醒为 HIGH，850ms 不会以同一 event_id 再次发出同等级的首次提醒。该行为可能消除了同目标重复，也可能在 tracker ID 不可靠时压制新目标提醒；现有数据不足以判别。

在 2,600ms 事件的相同创建签名下，Baseline 的 HIGH 首次提醒发生在 3,550ms，score 0.722081，带 `[29, 1]`；v2 的 HIGH 首次提醒发生在 2,850ms，score 0.816760，仅带 `[29]`，比 Baseline 早 700ms。实验策略让 track 1 与 track 29 的类别续接过程不同，因此关联来源和评估历史发生变化。该时刻提前不等于已证明风险发现更及时或更正确。

## 6. 失败测试与最小实验实现

### 6.1 先写测试的过程

开发先定义配置开启时的预期续接以及 Baseline 默认拒配行为，增加测试后观察 Baseline 因类别硬门控产生新事件；之后实现开关。第一轮实验回放又发现仅中心距离仍可能接受框几乎不重叠的跨类别匹配，因此先加入“低重叠跨类不续接”回归测试。未加 IoU 保护时该测试失败：中心接近但 IoU 低于 0.10 的 car 被错误续接到 truck 事件；加保护后按预期新建事件。

低 IoU 风险在初轮实验日志中体现为 V01 `bench→person`（IoU 0）和 V05 `person→chair`（IoU 0.048、IoU 0）候选。最终实验版不接受这些跨类续接。该单元测试验证的是明确配置策略和几何保护，不代表真实对象身份 Ground Truth。

### 6.2 实现内容

- 在 `AssociationConfig` 新增 `allow_class_switch_for_track_id: bool = False`，验证值必须为布尔类型。
- Baseline 风险配置显式写 `false`；新增 `configs/risk-association-v2.yaml`，保持其余风险与提醒配置相同，仅打开实验开关。
- 对实验跨类别匹配保留同 track ID 匹配，增加帧内检测 ID 唯一、活动事件当前 track ID 唯一、恢复时间窗、中心距离、IoU 和原有互为唯一最佳条件。
- 输出关联方法 `track_id_class_switch` 以便诊断，并把接受后的事件类别更新为当前检测类别。
- 不直接合并不同 track ID，不改变同类别几何策略。事件提醒历史保持原 event_id，不做跨事件继承。
- 在 `docs/research/phase2.md` 说明配置语义和边界；没有更改 `AGENTS.md`、默认风险阈值、提醒算法或空间音频。

### 6.3 新增测试覆盖

新增/扩展测试覆盖：实验开关类别续接、默认关闭时跨类别拒配、中心距离门控、低 IoU 拒配、重复 ID 歧义拒配、双目标交叉不合并、提醒历史不重置、配置布尔校验。既有测试继续覆盖 tracker ID 恢复、几何关联、会话隔离、生命周期和提醒调度。

测试只建立软件行为和不变量，不声称 tracker ID 与物理身份一致。

## 7. 同源离线回放方法

Baseline 与实验组都从 `outputs/real_world/v01/<video_id>/perception/tracks.jsonl` 重算风险帧、事件和提醒。未运行 YOLO，没有改变上游追踪结果和时间戳。Baseline 输出保存在 `outputs/algorithm_refinement_v1/baseline/`；最终 IoU 保护实验输出保存在忽略的 `outputs/algorithm_refinement_v1_iou_guard/experimental/`。

比较器 `scripts/compare_event_association.py`：

- 读取同一 manifest 下八段视频；校验 baseline/v2 使用的轨迹 SHA-256、源感知 metadata 哈希、配置、帧数和逐帧时间戳。
- 确认有效风险配置仅在 `allow_class_switch_for_track_id` 上有差异。
- 逐 observation 对齐关联方法、事件创建签名、关联前后状态和该时刻提醒记录。
- 统计事件数、首次/升级提醒、每分钟提醒、同帧竞争、跨事件 track 冲突、新事件冲突、疑似重复首次提醒候选、多 track 来源事件和关联调用耗时。
- 比较提醒记录的时间、等级、分数、来源 ID、质量分及原因，而非只比较提醒总数。
- provenance 记录 manifest/config/comparator 哈希及 `ground_truth_used=false`。逐视频 tracks 和源 metadata 哈希保存在 JSON 报告中。

最终报告在 `outputs/algorithm_refinement_v1_iou_guard/report_final/`：`comparison.json`、`summary.csv`、`association_cases.jsonl`。其中逐 observation 关联差异共 714 条；按事件创建签名计，49 个签名只在 Baseline 建立，10 个只在实验组建立。创建签名由创建时间、原始 track ID、初始类别组成，仅用于行为对齐，不是物体身份标注。初轮未加 IoU 的结果保留在 `outputs/algorithm_refinement_v1/`，用于解释为何增加 IoU 门槛，没有覆盖它。

## 8. 八段统计结果

### 8.1 事件和提醒

下表中 `B → v2` 表示 Baseline 到最终 IoU 保护实验策略。所有片段均为 15 秒，提醒频率按 manifest 视频时长计算。

| 视频 | 风险事件 | 首次提醒 | 升级提醒 | 总提醒 | 提醒/分钟 |
|---|---:|---:|---:|---:|---:|
| V01 | 100 → 97 | 31 → 32 | 2 → 2 | 33 → 34 | 132 → 136 |
| V02 | 123 → 121 | 51 → 51 | 4 → 4 | 55 → 55 | 220 → 220 |
| V03 | 84 → 77 | 32 → 31 | 2 → 2 | 34 → 33 | 136 → 132 |
| V04 | 36 → 35 | 18 → 17 | 2 → 2 | 20 → 19 | 80 → 76 |
| V05 | 140 → 129 | 33 → 34 | 3 → 3 | 36 → 37 | 144 → 148 |
| V06 | 65 → 64 | 22 → 22 | 3 → 3 | 25 → 25 | 100 → 100 |
| V07 | 121 → 109 | 40 → 38 | 8 → 8 | 48 → 46 | 192 → 184 |
| V08 | 60 → 58 | 5 → 5 | 1 → 1 | 6 → 6 | 24 → 24 |
| **总计** | **729 → 690** | **232 → 230** | **25 → 25** | **257 → 255** | **128.5 → 127.5** |

提醒总数整体仅变化 2 条，不代表提醒对象相同。逐事件提醒记录有新增、删除、时间、分数和来源变化，需结合具体案例与人工标注解释。

### 8.2 关联冲突、疑似重复与关联时间

| 视频 | 跨事件冲突观察 | 新事件冲突 | 重叠首次提醒事件对 | 多 track 来源事件 | 关联均值 ms/次 |
|---|---:|---:|---:|---:|---:|
| V01 | 56 → 41 | 10 → 7 | 0 → 0 | 9 → 9 | 0.0484 → 0.0716 |
| V02 | 45 → 34 | 5 → 3 | 1 → 1 | 20 → 20 | 0.0803 → 0.1103 |
| V03 | 329 → 40 | 13 → 1 | 10 → 0 | 22 → 13 | 0.0519 → 0.0692 |
| V04 | 119 → 45 | 6 → 3 | 4 → 2 | 8 → 4 | 0.0315 → 0.0380 |
| V05 | 142 → 104 | 32 → 22 | 2 → 2 | 25 → 23 | 0.0621 → 0.0879 |
| V06 | 108 → 96 | 2 → 1 | 2 → 2 | 8 → 7 | 0.0386 → 0.0504 |
| V07 | 88 → 3 | 14 → 1 | 1 → 0 | 18 → 15 | 0.0764 → 0.1020 |
| V08 | 36 → 30 | 7 → 5 | 0 → 0 | 10 → 10 | 0.0222 → 0.0290 |
| **总计** | **923 → 393** | **89 → 43** | **20 → 7** | **120 → 101** | **0.0514 → 0.0698** |

口径说明：

- “跨事件冲突观察”是在一帧 observation 所属事件之外，另有未结束事件的 `last_track_id` 或历史 `source_track_ids` 主张同一 track ID 时计数；它不是独立真实目标数量。
- “新事件冲突”是新建事件的 observation 同时与别的未结束事件分享 track ID 的次数。
- “重叠首次提醒事件对”是相同 track ID 对应两个生命周期相交、且两个事件都发过首次提醒的候选对；同一 ID 可能是合法换号轨迹，也可能是身份错误。
- “多 track 来源事件”是来源 track ID 数量大于一的风险事件。减少既可能来自更稳定的关联，也可能反映错误合并减少了来源分裂。
- 同帧一个 track ID 对应多个 observation/event 的结构性竞争，两策略均为 0。
- 关联耗时是在完整引擎之外单独计时 `EventManager.associate` 调用，每策略 2,400 次。均值与 p95 分别为 Baseline `0.0514ms/0.1232ms`、v2 `0.0698ms/0.1787ms`。本机单次计时受调度和环境影响，不等于模型端到端时延。

### 8.3 提醒变化风险

Baseline 风险等级提醒总量为 HIGH 120、MEDIUM 137；实验组为 HIGH 116、MEDIUM 139。按事件创建签名和类型比较，有 8 条 Baseline-only HIGH 与 4 条实验组-only HIGH。相同签名也可能发生时间、分数、提醒来源 ID 或原因变化；比较器最终报告共保留 29 行提醒记录差异。

主要 V03 HIGH 差异：

- `350ms|track 1|car`：Baseline E000004 在 850ms HIGH / 0.800 提醒，v2 未建立该事件，未再提示。Baseline 的 E000003 已在 250ms HIGH / 0.744 提醒。该差异有“同对象去重”和“新对象被压制”两种可能解释。
- `2600ms|track 29|motorcycle`：Baseline 3,550ms HIGH / 0.722，sources `[29, 1]`；v2 2,850ms HIGH / 0.817，sources `[29]`。时刻提前 700ms，但事件更新路径和来源不同。
- 其他创建签名下也有 10,400ms、10,950ms 的 Baseline-only HIGH 与 7,650ms、10,400ms 的实验组-only HIGH。完整提醒 JSON 保留其 track/event、分数和时间。

总 HIGH 少 4 条和 V03 的具体差异应作为人工审查队列，而不能解释为漏报或误报结论。输入片段没有独立标注，不能判定任何一个提示是否对应真实重要风险。

## 9. 开发验证与交付文件

### 自动化验证

- 定向回归：`tests/test_decision_phase2.py tests/test_risk_phase2.py tests/test_alert_diagnosis.py`，39 passed。
- 完整 pytest：`uv run pytest -q --basetemp models/hybrid/tmp/pytest-algorithm-refinement-v1`，354 passed、5 skipped，19.31 秒。
- `uv run ruff check .`：通过。
- `uv run ruff format --check .`：通过，159 files already formatted。
- `git diff --check` 和提交前 `git diff --cached --check`：通过。
- 八段离线回放：每段 300 帧，无模型推理；比较器检查输入轨迹 hash、源 perception metadata hash、帧数、时间戳和两策略配置差异。

### 本次实现提交包含的文件

- `src/sightguard/decision/events.py`、`src/sightguard/risk/config.py`：实验关联开关、检查条件及关联方法标记。
- `configs/risk.yaml`、`configs/risk-association-v2.yaml`：默认关闭配置和单变量实验配置。
- `tests/test_decision_phase2.py`、`tests/test_risk_phase2.py`：关联、保护条件和配置测试。
- `scripts/compare_event_association.py`：同源轨迹差异、提醒记录差异和关联耗时比较。
- `docs/research/phase2.md`、`docs/research/open_source_algorithm_review.md`、`docs/research/algorithm_refinement_v1.md`：策略边界、开源调研和简要结果报告。

本详细报告作为后续追加文件单独生成。原始视频、缓存、实验输出和两份用户既有未跟踪文档未进入提交。

## 10. 复现步骤

请选用从未存在的新输出路径；`sightguard assess` 会拒绝覆盖已有 assessment artifacts。以下命令只重算保存轨迹上的风险、事件和提醒：

```powershell
$root = 'outputs/algorithm_refinement_v1_replay'
$manifest = Get-Content -Raw datasets/real_world/manifests/dataset.json | ConvertFrom-Json
foreach ($video in $manifest.videos) {
    $tracks = "outputs/real_world/v01/$($video.video_id)/perception/tracks.jsonl"
    $output = "$root/experimental/$($video.video_id)"
    uv run sightguard assess --tracks $tracks --config configs/risk-association-v2.yaml --output-dir $output --stream-id $video.video_id
    if ($LASTEXITCODE -ne 0) { throw "Assessment failed: $($video.video_id)" }
}
uv run python scripts/compare_event_association.py `
  --manifest datasets/real_world/manifests/dataset.json `
  --tracks-root outputs/real_world/v01 `
  --baseline outputs/algorithm_refinement_v1/baseline `
  --experimental "$root/experimental" `
  --baseline-config configs/risk.yaml `
  --experimental-config configs/risk-association-v2.yaml `
  --report-dir "$root/report"
```

输出格式与逐帧示例见 `outputs/algorithm_refinement_v1_iou_guard/report_final/`：`comparison.json` 含逐视频 provenance、汇总及 alert deltas；`summary.csv` 为逐视频策略指标；`association_cases.jsonl` 为 714 条 observation 关联差异。生成数据位于 Git 忽略目录。

## 11. 局限与下一步

目前没有被确认且可以称为“已经修复”的真实身份错误。实验开关实现了类别抖动下的一种保守续接方案，并确保默认输出兼容，但身份正确性仍未验证；即使 exact track ID 唯一、框有 IoU 重叠，也不能保证 tracker ID 对应同一实体。

最值得先做的是盲标 V02 的 E000006/E000020 片段和 V03 track 1 的 0–4 秒、2.6–3.7 秒片段，同时抽查其他 Baseline-only/实验组-only HIGH 提醒。标注至少应区分物理对象身份、遮挡/短时不可见、tracker ID 切换、是否为重要风险事件及提醒机会时间。随后在冻结的同一 tracks 上重放两策略，对 false merge、独立目标 HIGH 提醒保留情况与提示延迟单独评价，再决定是否保留、修改或撤销实验策略。

在得到复核标签前：不应启用 v2 作为默认策略，不应以提醒减少证明改进，也不应把模拟测试或回放统计表述为碰撞预测、道路安全保证或风险检出率。
