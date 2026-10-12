# Phase 2：因果风险评估与事件级提醒

本阶段是二维图像启发式研究实现。R 是相对评分，Q 是感知质量指标，二者都不是经过标定的
概率。没有绝对距离、物理接近速度、TTC、真实行进路径或道路安全保证。

## 数据流与职责

schema 1.0 FrameRecord → 空间框校验 → EventManager 一对一关联 → FeatureExtractor →
RiskScorer → 生命周期更新 → PerceptionMonitor → AlertScheduler → schema 2.0 StepResult。
`AssessmentEngine` 持有单视频所有状态；相邻视频必须新建实例。纯核心仅用 Python 标准库，
配置解析使用 PyYAML。CLI assess 流式读取，不加载任何视觉模型/视频。
事件生命周期与提醒历史分开；提醒后事件仍可 ACTIVE/LOST，不因提醒而结束。

## 特征定义

空间特征使用原始像素框除以当前帧宽高，裁剪至 [0,1]。宽/高/面积为归一化图像值，中心
`((x1+x2)/2,(y1+y2)/2)`，底部中心 `((x1+x2)/2,y2)`。零面积或完全在画面外的框无法评估：
该 detection 输出 error、features/assessment/event_id 为 null；一帧有这种框即标为 partial/limited，
不把未出现的其他目标当作已消失。部分越界的有效框裁剪，并输出 bbox_clipped 和质量限制标记。
Phase 1 已拒绝倒置/非有限框；这类原始 JSONL 格式错误导致 assess 失败，而不是伪造合法帧。

关注区域为归一化梯形：中心 x=0.5，顶 y=0.35、半宽 0.12，底 y=1、半宽 0.45。
底部中心在梯形内时 S=1；外部取到梯形边的最近欧氏距离 d，S=clip(1-d/0.15,0,1)。
全部位置/宽度/软边界参数均可配置。该区域只是摄像画面的近似，不是已估计的行进路径。

时间特征以内部 event_id 为关联键，原始 ID 切换可保留历史：

- observation_duration_ms：当前时间减该事件首个有效观察时间；含中断，不代表连续可见时长。
- observation_interval_ms：距上次有效观察的实际毫秒间隔，第一次为 null。
- motion_interval_ms：本次运动计算使用的基线间隔，与上一观察间隔可能不同。
- scale_change_per_s：`log(sqrt(area_now/area_baseline))/(dt_ms/1000)`，可正可负，单位为图像相对尺度/秒。
- center_change_per_s：归一化中心位移除以实际秒数，输出 x/y 分量；不是世界坐标速度。
- inward_change_per_s：距配置中心 x 的绝对偏差减少量除以秒数。
- detection_stability / continuity：近期有效观察次数除以窗口内可用观测机会数；失败/partial 帧
  对未观察目标是 unknown，不作为负观察计入分母。

默认窗口 1500ms。有效运动基线间隔在 10–750ms；短间隔先累计，不会让高帧率永远缺运动信息。
尚未达到基线间隔的当前样本输出运动 null，不伪装成静止。成功计算后更新基线；重复时间戳、
首次观察、中断或过大间隔时不给不可计算变化率补零。遮挡/失败后首个恢复样本重置运动基线，
下一段有效连续观察才恢复运动证据。运动率可用因果 EMA，默认 tau=150ms，alpha=1-exp(-dt/tau)，
tau=0 禁用。绝不读取未来帧平滑过去。

Q 单独计算：`0.5*confidence + 0.3*continuity + 0.2*min(1,recent_observations/4)`，
缺 ID 乘 0.85、中断乘 0.75、FPS 回退时间乘配置的 0.8，最终裁剪至 [0,1]。
系数是当前算法定义，不是可靠性标定；观察数量目标和回退因子可配置。质量输出还包含累计有效
观察数、近期观察数、ID 缺失、interrupted、motion_reliable、时间来源和具体 limitations。
**Q 不乘 R，低检测置信度不会直接把高图像风险变成低风险。**

## 评分、降级与迟滞

所有 clip 都是 [0,1] 线性饱和；归一化范围和混合权重从配置读取。

```text
S = focus_relevance
area_score = clip(area / area_full_score)
bottom_score = clip((bottom_y-bottom_start)/(bottom_full_score-bottom_start))
A = appearance_area_weight*area_score + (1-appearance_area_weight)*bottom_score
growth_score = clip(max(0,scale_change_per_s)/scale_rate_full_score)
inward_score = clip(max(0,inward_change_per_s)/inward_rate_full_score)
M = motion_scale_weight*growth_score + (1-motion_scale_weight)*inward_score
R_raw = spatial_weight*S + appearance_weight*A + motion_weight*M
```

默认 R 权重 0.45/0.35/0.20；A 面积混合权重 0.70，面积满分 0.20，底部范围 0.30–0.95；
M 尺度混合权重 0.75，尺度满分变化率 0.50/s、向中央满分变化率 0.25/s。
中心 x/y 变化均导出，但 M 第一版只使用尺度增长和向中央的 x 变化，不将任意横/纵移动视为接近。
M 缺失时为 null，R=`(ws*S+wa*A)/(ws+wa)`，明确 degraded=true 与解释字符串。
这会使缺证据的分数相对已知静止状态升高，可能产生保守误报；缺失不是零运动，也不是安全。
有充分静止证据时 M=0，S/A 仍保留风险。当前没有逐类别物理风险权重。

可选分数 EMA 默认 tau=0（禁用），启用时以视频 dt 计算 alpha。首个样本无历史状态。
初始等级 LOW <0.35、MEDIUM [0.35,0.65)、HIGH ≥0.65；已有等级升级需超过边界+0.02。
降级需低于当前边界-0.05，并在有效观察中持续 400ms。回到迟滞带或中断即重置降级计时。
输出 score 是平滑分数，raw_score 是加权原分数；等级可能因迟滞不同于即时分数所在区间，
原因包括 risk_level_held_by_hysteresis。无观察和失败帧不修改历史评分、也不推动风险降级。

## 关联与状态机

候选按 `(class_id,normalized_bbox,track_id_or_-1,-confidence,input_index)` 稳定排序。
新事件顺序分配 E000001 等 ID。关联先尝试仍有效的当前原始 track_id：类别相同，ID 在两端
均唯一，距上次观察不超过 750ms，归一化中心距离≤0.35。只匹配当前 ID，不拿历史 ID 强行合并。
随后对未配对目标尝试几何：同类别、同恢复窗口、中心距离≤0.15、IoU≥0.10，评分
`0.60*IoU+0.40*(1-distance/0.15)`≥0.35。
两端都要求唯一最佳，最佳与次佳分差≤0.08 视为歧义并拒绝关联；排序按评分降序、event_id、
detection_index 决胜。每帧一个事件最多关联一个目标，一个目标最多关联一个事件。
有歧义时原事件暂时 LOST、新目标新建候选；宁可分裂，也不强制合并无法解释的身份。

`association.allow_class_switch_for_track_id` 默认为 `false`，维持上述 Baseline。实验配置
`configs/risk-association-v2.yaml` 将其开启：仅当当前和历史 `track_id` 在帧内与活动事件中均唯一、
视频时间处于恢复窗口内、中心距离不超过 0.35 且框 IoU 不低于现有 `geometry_min_iou`（0.10）时，
才允许相同 tracker ID 跨检测类别续接。
几何匹配仍要求类别相同；冲突或越过距离门限时照常拒绝。续接记录的
`association_method="track_id_class_switch"` 标明类别发生变化，事件快照更新为当前检测类别。
提醒历史仍按原 event_id 保留。此策略不合并不同 track_id，也不证明追踪器 ID 对应同一真实目标；
它仅用于研究稳定 tracker ID 与检测类别短时波动对事件分裂的影响。

创建 → CANDIDATE；满足 250ms 与至少 2 次有效观察 → ACTIVE。
候选确认区间中相邻有效观察间隔不得超过 500ms；成功空帧或未知帧使未观察候选重新计时。
有效观察次数另行累计，不用墙钟等候来补确认条件。
完整成功帧上未观察到 → LOST；750ms 窗口内匹配恢复，曾确认者恢复 ACTIVE，未确认者恢复
CANDIDATE 并重新确认。成功完整帧到达且距离 last_seen 超过窗口 → RESOLVED。
RESOLVED 终态，即便原 tracker ID 复用也新建事件。

失败/有无效检测的 partial 帧不使未观察事件 LOST/RESOLVED；observed_now=false、历史评分
保持，但不根据过期快照发场景提醒。恢复后的完整帧才按真实经过的视频时间应用匹配/失效窗口，
长时间未知后可能新建事件；RESOLVED 在这里表示关联观测过期，不表示物理障碍已消失。
每条事件更新记录 raw source_track_id、内部 event_id、关联方法、累计 source_track_ids；null 原样保留。
输入结束只输出 censored 日志，保持未决状态，不延长视频时间、自动制造 RESOLVED 或新提醒。

## 提醒与感知状态

普通事件只有 ACTIVE、observed_now=true、风险达到配置 minimum_level（默认 MEDIUM）才可首次提醒。
默认 repeat_enabled=false：不对风险等级未变的事件重复提醒。开启后普通间隔为 3000ms。
有意义升级需等级超过该事件已提醒过的最高等级、score 增加至少 0.05、距上次提醒≥1000ms；
该间隔独立于普通冷却，若条件尚未满足会在后续当前观察中再次评估。
降级后回到已经提醒过的最高等级默认不再升级提醒。

多目标排序：等级降序 → S 降序 → score 降序 → event_id 字典序。每帧最多 1 条提醒，可配置；
未获得发送名额的事件不标记为已提醒，后续有效帧仍可候选。
提醒只输出 dataclass/JSONL，不访问音频硬件。

感知状态 available/limited/unavailable 独立于风险：错误帧为 unavailable，无效框、重复时间戳、
过大帧间隔或目标 Q<0.30 为 limited。持续 500ms 生成 H000001 等独立事件和 perception_limited
提示（risk_level/risk_score 为 null）；一个连续受限片段默认提示一次，恢复 available 时结束。
这种提示优先于场景风险提示，占用同一每帧名额，但未抢到名额的场景事件可后续发送。
一次孤立的最终失败帧时，只有候选健康事件和截尾记录，未观测到足够持续时间就不伪造持续失败提示。

## 时间异常、输入与会话

输入帧必须从 0 连续编号，时间有限、非负、非递减。缺失/非法/倒退时间戳直接报错并保留 failed
元数据及已输出的前缀。相等时间戳允许，但标记 limited、运动不可用，不能因此跨越确认/冷却时间。
超过 1000ms 的单帧间隔标记 limited，可配置；默认 750ms 关联窗口/500ms 确认间隔不适合低于
约 2FPS 的输入，长间隔不会凭空确认目标，应调整配置并记录。
时间来源沿用 Phase 1 video_pts/fps_fallback，回退精度的局限不在这里消失。
事件有效时长/提醒间隔都来自这些视频时间；墙钟仅用于元数据审计。

stream_id 默认由输入绝对路径哈希产生，可用 --stream-id 指定稳定标识。输入全文 SHA-256 只在
元数据记录；不能用未来内容改变过去事件标识。事件/提醒编号在每视频重置，跨视频引用需带
stream_id 与运行来源。相同输入与配置下三种业务 JSONL 逐字节一致；审计元数据的时刻和耗时可不同。

## 输出 schema 2.0

每条业务 JSONL 根对象都有 schema_version="2.0" 和 stream_id：

- risk_frames：frame_index/timestamp_ms/timestamp_source/input_status/perception/observations/events/warnings。
  observations 保持原 detection_index，包含 source_track_id/class_id/class_name/event_id/association_method/
  features/assessment/error；无效框的评估为 null。events 是当前未结束事件的显式 observed_now 快照。
- events：event_id/timestamp_ms/event_kind/action/state/reason/source_track_id/association_method/snapshot。
  action 为 created/updated/activated/lost/recovered/resolved/censored；event_kind 为 risk/perception，
  感知事件没有风险目标 snapshot。censored 是观测结束日志，不是第五种事件状态。
- alerts：alert_id/event_id/timestamp_ms/risk_level/risk_score/alert_type/reason/source_track_ids/quality_score。
  alert_type 为 first/escalation/repeat/perception_limited；perception_limited 没有虚构的场景评分。
- run_metadata（单 JSON）：算法版本、输入/源码/有效配置哈希、Python/包/PyYAML 版本、时间/统计/状态。
  source_run_status 与 source_error_frames 显式保留源运行完整性；assess completed 只表示这份数据
  被分析完成，不表示原感知完整成功。错误时保留部分结果，status=failed；已有输出拒绝覆盖。

完整类型定义位于 risk/schema.py，手工预期案例见 tests/fixtures/phase2/static_occlusion.expected.json。
当前仅导出结构化结果，未改动 Phase 1 标注视频，也未开发 UI、音频播放、三策略对照或独立标注平台。

## 已知局限与 Phase 3 验证问题

相机运动、转身、目标姿态变化均可能使尺度/位置变化被误认为动态证据；小框/遮挡/低光会使
关联分裂或错配；近距离同类交叉且 tracker 仍报告唯一但错误的 ID 时，几何门控不足以识别所有 ID 交换。
保守拒绝几何歧义会新建候选，可能重复提醒同一物理目标，也可能因反复确认而漏提醒。
视场外、漏检、无效低帧率输入仍可能漏掉重要风险；空检测不代表环境安全。
缺运动权重重分配、迟滞和确认时间都有误报/延迟/漏检权衡。健康提示默认最高优先级会推迟场景提示。
长期运行保留已结束事件档案，超长输入的内存开销尚未做性能基准。

Phase 3 必须用独立事件标注建立真实动态场景样本、统一相机/类别/时间处理约定、划分调参与测试集，
验证重要事件检出、重复/无效提醒、提醒延迟、ID 切换/歧义以及感知失败的影响，并复用同一感知 JSONL
比较策略。模拟轨迹正确性不构成真实道路安全验证，当前没有可宣称的准确率或安全性数据。
