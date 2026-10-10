# SightGuard Algorithm Refinement v1

检查日期：2026-10-11。范围限于事件身份关联；未重跑 YOLO、未更改风险阈值、默认提醒或空间音频。八段 UrbanEgo 视频均来自同一次原始拍摄，总时长 120 秒、2,400 帧，尚无已复核的事件 Ground Truth。

## 结论

代码证实了一个明确的行为机制：当前 EventManager 将类别相同作为 track_id 续接的硬条件。V03 中 track 1 的检测类别从 truck 变为 car 时，Baseline 新建了 E000004；之后它又以原 track_id 间歇恢复。这个规则会在类别抖动时拆分跟踪事件，但现有证据不能判定相关事件是否对应同一真实目标，因此没有把它称为已确认的身份错误或安全缺陷。

实现了一个默认关闭的 `allow_class_switch_for_track_id` 实验策略。它减少了多项基于 track_id 的事件关联冲突，但八段回放出现 HIGH 提醒被抑制、事件归属和提醒时刻改变等情况。没有独立标注能区分合理去重与错误合并，故结果不足以证明实验策略整体改善；Baseline 仍是默认策略。

## 关联与提醒的代码行为

- 类别相同是原有 track_id 与几何关联的共同条件；同一帧及活动事件中的 track_id 竞争由互斥的一对一匹配处理。关联失败时可能新建事件。
- AlertScheduler 的提醒历史按 event_id 保存。LOST 恢复不清除该历史；事件新建会形成新的提醒历史。
- Video 02 中 E000006 与 E000020 的首次提醒分别发生在 600ms 和 1850ms，二者时间段重叠。E000006 于 1550ms 从 track 9 通过几何关联转到 track 6；E000020 于 1450ms 从 track 94 创建，1650ms 通过几何关联转到 track 9。两个事件在 1850ms 都仍活动，E000020 当时发出 MEDIUM 首次提醒。v2 在这些帧的事件、关联、提醒时刻与 Baseline 相同；跨事件冲突观察从 45 降至 34，但该目标身份案例没有修复。无法判断 9、6、94 是否对应同一人或不同人。
- Video 03 中 E000003 在 250ms 对 track 1 产生 HIGH 首次提醒（score 0.743952）。类别变更后，Baseline 于 350ms 为同一 tracker ID 创建 E000004，并于 850ms 产生 HIGH 首次提醒（score 0.800000）。v2 在 350ms 以 `track_id_class_switch` 将该观察续接到 E000003；因提醒历史属于 E000003，850ms 没有第二次 HIGH 提醒。若类别变化仍是同一对象，这是可能的重复抑制；若对象身份已错，可能压制新目标提醒。缺少人工标注，不能选择其中一种解释。
- Video 03 的 2,600ms 事件创建签名（track 29、motorcycle）在两策略中都有一条首次 HIGH 提醒，但行为不同：Baseline 于 3,550ms 发出 score 0.722080 的 HIGH，提醒携带 track 29、1；v2 于 2,850ms 发出 score 0.816760 的 HIGH，仅携带 track 29。该签名下提醒提前 700ms，关联来源与分数也变化。物理身份和提前提醒是否有益均未验证。

## 实验策略与保护条件

`AssociationConfig.allow_class_switch_for_track_id` 默认 `false`；`configs/risk.yaml` 显式关闭。独立的 `configs/risk-association-v2.yaml` 只将此开关设为 `true`。启用时，跨类别续接还要求：

- track_id 在当前有效检测和活动事件的当前 track_id 中均唯一；
- 视频时间差不超过现有 750ms 恢复窗口；
- 中心距离不超过现有 0.35；
- 框 IoU 达到现有 `geometry_min_iou`（0.10）；
- 原有互为唯一最佳、一对一匹配及歧义拒配规则成立。

接受的跨类别续接以 `track_id_class_switch` 记录，并把事件类别更新为当前检测类别。提醒历史仍属于原 event_id。新增逻辑没有把不同 track_id 直接当作同一目标；同类别几何关联规则未变。tracker 错误复用或维持错误 ID 仍可能使不同真实目标合并，现有视觉框条件不能证明实体身份。

最初的跨类别试验曾出现中心距离合格但几乎没有框重叠的匹配，包括 V01 `bench→person`（IoU 0）、V05 `person→chair`（IoU 0.048 与 0）。所以最终实验版在跨类别续接上增加最小 IoU 限制；低重叠回归测试在加入此条件前失败、加入后通过。这个额外条件仅约束新跨类别路径，不改变默认算法。

测试先验证类别开关关闭时 Baseline 的拒配行为、开关启用时策略续接行为，再覆盖距离、IoU、歧义、多目标交叉和提醒历史。它们证明程序遵守配置和不变量，不证明同一 track_id 对应同一真实目标。

## 同源离线回放结果

Baseline 为当前默认配置；实验组仅启用上述开关与 IoU 保护。使用原有八份 `perception/tracks.jsonl` 重算下游事件和提醒，没有运行模型。比较器检查两组 tracks SHA-256、源感知 metadata 哈希、帧数、时间戳及有效配置；两组输入哈希一致。

- 风险事件数：729 → 690。
- 提醒：257（首次 232、升级 25）→ 255（首次 230、升级 25）。每分钟提醒：128.5 → 127.5。
- 全部风险等级提醒中，HIGH 为 120 → 116，MEDIUM 为 137 → 139。按事件创建签名分组，有 8 条 Baseline-only HIGH 与 4 条实验组-only HIGH；它们是需要复核的行为差异，不代表已确认漏报或误报。
- 含同一 track_id 跨多个活动事件的观察：923 → 393；其中在 `new` 事件上出现另一活动事件仍主张同 ID 的冲突记录：89 → 43。
- 同帧一个 track_id 被多个 observation 分配到不同事件：0 → 0。
- 同一 track_id 关联多个同时存活、且各自曾产生首次提醒的事件对：20 → 7。仅为疑似重复候选；不等同于物理目标重复。
- 事件曾包含多个 track_id 的数量：120 → 101。该数可能同时来自合法 tracker ID 切换和错误合并。
- 关联调用计时：每种策略 2,400 次。均值 0.0514ms → 0.0698ms；p95 0.1232ms → 0.1787ms。计时来自本机一次离线调用测量，受运行环境影响，不代表完整模型推理或设备性能基准。

V02：123 → 121 个风险事件；首次提醒 51 → 51、升级 4 → 4、总提醒 55 → 55；跨事件 track 冲突观察 45 → 34，新事件冲突 5 → 3。已知 E000006/E000020 案例的 600ms / 1850ms 两次提醒及 9→6、94→9 的事件迁移未改变。

V03：84 → 77 个风险事件；首次提醒 32 → 31、升级 2 → 2、总提醒 34 → 33；跨事件 track 冲突观察 329 → 40，新事件冲突 13 → 1；疑似重叠首次提醒事件对 10 → 0。除上面 850ms 案例外，按事件创建签名计的 Baseline-only HIGH 提醒还包括 10,400ms 与 10,950ms 两项；实验组另有 7,650ms 与 10,400ms 的 HIGH 提醒。提醒下降不能单独作为策略收益。

完整逐视频指标见生成的 `summary.csv`；事件创建签名差异、提醒记录/时间差和逐 observation 关联状态见 `comparison.json` 与 `association_cases.jsonl`。这些生成数据位于忽略的 `outputs/algorithm_refinement_v1_iou_guard/report_final/`，不进入 Git。较早无 IoU 保护的试验记录保留在 `outputs/algorithm_refinement_v1/`，用于审计为何增加 IoU 限制。

## 开源参考

研究记录在 [open_source_algorithm_review.md](open_source_algorithm_review.md)。CaBot People 的关联实现使用 3D 鸟瞰中心、预测与 Hungarian assignment；其传感器假设不适合直接移植到当前单目框管线。Roana 的已实现代码处理深度网格与走廊选择，不维护 SightGuard 所需的 track/event 身份或首次提醒历史。没有复制代码；记录包含仓库版本、源码路径、传感器需求和许可说明。

## 验证与限制

本次定向回归已覆盖 `tests/test_decision_phase2.py`、`tests/test_risk_phase2.py`、`tests/test_alert_diagnosis.py`。离线回放只验证原感知结果上的下游软件行为，没有真实推理、独立 Ground Truth 或安全性评价。任何 HIGH 提醒的移除、提前或来源变化，仍需逐帧人工标注来判断是同一目标去重还是身份错误。

复现最终 IoU 保护试验（请选择一个新的输出根目录；assessment 会拒绝覆盖既有文件）：

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
