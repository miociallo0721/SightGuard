# Phase 3 方法学与实验协议

## 研究问题与边界

本阶段比较三种提醒策略：在完全相同的 Phase 1 感知帧记录上，完整的 SightGuard 决策能否在保留重要风险事件检出的同时，减少无效提醒和不必要的重复提醒。人工真值来自原始视频，由标注者在不查看检测框、跟踪 ID、风险分数或策略提醒的条件下独立标注。

本实验评价的是普通视频中可见的画面风险线索和提醒机会，不估计真实距离、碰撞概率或真实行走轨迹，也不构成道路安全保证。当前仓库没有已审核的真实动态视频标注及真实数据评价结果；合成用例只验证程序逻辑，不代表真实性能。不得将离线重放耗时表述为移动端或在线端到端延迟。

## 数据和盲化流程

每个 `video_id` 对应一份原始视频、一份 Phase 1 `tracks.jsonl`、一份独立标注 JSON 和 manifest 记录。三种策略按同一份帧记录和视频时间戳顺序执行；中间数据不得因策略不同而重新推理。重要目标即使未出现在 `tracks.jsonl` 中，只要人工真值中存在且可评价，仍保留为漏检。

真实标注必须以原始视频为依据，标注时隐藏感知输出和所有策略结果。建议由另一位人员复核，分歧按标注规范记录并处理。程序中的路径、来源字段和哈希只能验证文件身份及实验后是否改变，无法证明标注者实际上没有看过算法输出，也无法证明不存在人为偏差；盲化和复核过程应另行留存记录。

`datasets/manifest.csv` 中的 `source_group_id` 表示同一原始拍摄序列或场景来源。原视频及其裁剪片段应整体进入 development 或 test 的一个划分。校验会检查跨划分的来源组、原视频 SHA-256 和感知文件 SHA-256；这些检查不能发现所有重新编码、近似画面或未正确填写来源组的重复内容，仍需人工检查。development 用于参数和规则选择；test 仅在协议冻结后运行，不得根据 test 结果改参数或标注。

## 三种策略

- **A：基础目标触发。** 对配置中的固定类别和前方关注区域作逐帧判断，满足条件即产生 `trigger` 提醒。使用固定严重程度标签，不使用事件级确认、冷却或重复抑制。此项是简单端到端基线；A 与 C 的差异不能全部解释为事件管理的单独效果。
- **B：固定阈值和冷却。** 使用共享 Phase 2 风险特征与评分，达到固定风险分数阈值时触发；每个视频使用固定冷却间隔，间隔内不发下一条场景提醒，并按风险等级、空间分数和风险分数选择优先目标。B 不使用 C 的提醒升级及事件级重复提醒调度，但为保持风险特征口径一致，仍消费同一 Phase 2 评分流。
- **C：SightGuard。** 直接使用现有 Phase 2 `AssessmentEngine` 的风险评分、事件关联和生命周期、候选确认、迟滞、首次提醒、升级提醒、重复抑制、多目标优先级以及独立的感知受限提醒。不得用实验专用的简化实现替换该策略。

仓库当前 benchmark 配置初值为：A 使用类别 `[0, 1, 3]`、关注相关性阈值 `0.8`、固定 `MEDIUM` 严重程度；B 使用风险分数阈值 `0.35`、全视频冷却 `3000 ms`；匹配 IoU 下限为 `0.10`，重要事件等级默认为 `MEDIUM` 和 `HIGH`。这些是待 development 数据验证的工程初值，不是已验证的最优参数。任何调整都应记录在 benchmark 配置及冻结协议中。

三种策略使用相同 `FrameRecord`、相同视频时间戳和相同感知结果；决策只使用当前帧及此前状态。每次运行记录策略配置、风险配置、数据文件哈希、源代码指纹、划分及结果。处理失败、标注无效或输入缺失必须以失败/不可评价状态呈现，不能改写为空场景或零提醒。

## 标注和评价规则

标注中的每个已确认、可提醒事件必须提供一个 `initial` 提醒机会；可以没有 `escalation` 机会，也可按独立标注提供多个升级机会。机会的 `window_start_ms` 到 `deadline_ms` 两端均有效，告警需满足机会类型、最低严重程度和目标类别条件。机会窗口必须在事件时间范围内。标注窗口应在执行策略前依据原始视频和固定规则制定，禁止根据待测算法输出反推窗口。

匹配按视频独立进行，不使用算法自己的 `event_id` 或 BoT-SORT `track_id` 建立真值身份。候选提醒必须满足视频、类别、严重程度、提醒类型、包含两端的机会时间窗，以及存在有效目标区域时的空间重叠条件。算法的 `first` 只匹配 `initial`，`escalation` 只匹配 `escalation`；基线的 `trigger` 和 `repeat` 可匹配任一尚未占用且类型条件兼容的机会。目标区域使用标注的归一化框，IoU 下限由匹配配置预先确定。

对提醒与机会构成的二分图执行确定性的最大基数一对一匹配：一条提醒最多匹配一个机会，一个机会最多接收一条提醒。机会按窗口起点、截止点、真值事件 ID 和机会 ID 排序；提醒按视频时间、帧号和提醒 ID 稳定排序，使用增广路径求最大匹配。多个同类风险事件在时间上重叠时，标注应提供目标区域以区分对象；如果同类重叠事件缺少可用区域，含糊的提醒不强行归属，记为无效提醒并留下歧义原因。

机会匹配完成后，未匹配提醒若唯一落在已有匹配的同一人工事件内、且目标身份条件成立，则标记为重复提醒。无新机会支持的重复提醒及其他未获真值支持的场景提醒都计入 FP；重复提醒是 FP 的子集，不重复另加一次。不能唯一归属的提醒按歧义/不支持原因记录。`perception_limited` 是诊断类提醒，不作为场景 TP 或 FP，但在总提醒频率中单独计入。

## 不确定片段、排除与截尾

明确的不确定事件、标为 `excluded` 的事件和 `excluded_intervals` 都构成排除时间。排除区间按半开区间 `[start_ms, end_ms)` 判断提醒是否位于其中；重叠或首尾相接的排除区间合并后只扣除一次。任何与排除区间有正时长交集的已确认事件，整段事件及其机会都作保守删失；仅边界相接不触发删失。该事件的完整时间范围继续扩展排除并集，重复此过程直至没有新事件被删失，可能连带删失其他重叠事件。排除时段中的策略提醒标为 `ignored`，不作为 TP、FP 或总提醒频率；删失事件不进入事件或机会召回分母。

有效评价时长等于视频总时长减去合并后排除区间的总长度。正常、已审核且没有风险事件的视频是有效负样本；空检测帧不会使视频变成无风险样本。真实感知输入必须有成功完成的 Phase 1 元数据。数据缺失、源感知失败、标注仍为 draft 或策略执行失败都不应报告为成功的零提醒结果。

## 数据集和冻结流程

建议使用如下命令工作流；命令选项按 Phase 3 CLI 接口使用：

```powershell
uv run sightguard import-video --video path/to/video.mp4 --tracks path/to/tracks.jsonl --dataset-dir datasets --video-id clip-001 --source-group-id recording-001 --scene sidewalk --split development
uv run sightguard seal-labels --dataset datasets/manifest.csv
uv run sightguard validate-labels --dataset datasets/manifest.csv
uv run sightguard freeze-protocol --dataset datasets/manifest.csv --config configs/benchmark.yaml --risk-config configs/risk.yaml --output outputs/protocol.json
uv run sightguard benchmark --dataset datasets/manifest.csv --split development --config configs/benchmark.yaml --risk-config configs/risk.yaml --data-kind real --output-dir outputs/benchmark-development
uv run sightguard benchmark --dataset datasets/manifest.csv --split test --config configs/benchmark.yaml --risk-config configs/risk.yaml --protocol outputs/protocol.json --data-kind real --output-dir outputs/benchmark-test
```

`import-video` 只建立 draft 标注模板，不是负样本。人工完成标注并把文档状态改为 `reviewed` 后，先用 `seal-labels` 更新标注文件哈希，再用 `validate-labels` 验证 manifest、文件哈希、时间覆盖、元数据和标注结构。调参只在 `development` 划分进行。运行 test 前，使用最终标注、manifest、benchmark 配置、Phase 2 风险配置和当前代码冻结协议；test benchmark 必须提供该协议。协议会比较 manifest、输入哈希、配置和源代码指纹，发现变化时拒绝运行。哈希证明冻结后文件未变，不证明冻结前是否有人看过 test 结果或标注是否真正独立。

`--data-kind real` 与 `--data-kind synthetic` 应分开运行、分开报告；合成结果不能标为真实动态街景。图表默认开启，可在 benchmark 配置中关闭。每次 benchmark 应写入独立结果目录，不覆盖历史产物；至少保存指标、逐视频结果、匹配/错误案例、时间线和运行元数据。若样本数量或独立来源组不足，不给出伪精确的统计显著性结论；Bootstrap 如启用，应以视频或独立拍摄组为重采样单位，而不是连续帧。

本阶段不实现 Gradio UI、iOS 应用或新模型训练。Phase 2 公式、限制及状态机详见 [Phase 2 说明](phase2.md)。

## 当前实验口径的限制

GT 的 `event_type` 描述人工情境；策略当前不预测同名风险情境，因此匹配只使用人工
class_ids、目标区域、提醒类型、严重程度及窗口，不能据此声称验证了横穿/障碍等情境分类。
区域在某时刻缺失时，该时刻不能作为该事件的空间匹配证据；同类重叠事件应覆盖所有提醒窗口。
A/B 的感知受限诊断使用同样的健康监视器，但独立于场景触发；C 按核心调度器的优先级和
每帧容量统一调度。诊断提醒单独统计，不能把所有策略差异都归因于事件抑制。

冻结协议还绑定 Python/项目版本、pyproject.toml、uv.lock、主要依赖版本及真实 Phase 1
run_metadata.json 哈希。结果目录的 status=completed 且 artifacts_complete=true 才可复用；
失败时原始流保留为 .jsonl.partial，并记录失败视频/策略，禁止把部分结果当完整实验。
元数据的时间和墙钟耗时应随运行变化；确定性承诺针对指标、提醒、匹配和因果风险流，
不要求 Matplotlib SVG 随机元素 ID 或审计时间逐字节相同。
