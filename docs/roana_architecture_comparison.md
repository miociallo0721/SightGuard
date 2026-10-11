# Roana / SightGuard 架构对照

审计日期：2026-10-11。本轮仅研究架构和实现，不移植源码，不把上游开发状态视为道路可靠性证明。

## 1. 固定来源与状态

- 上游：[floatinglifeai/roana](https://github.com/floatinglifeai/roana)。本地审计固定提交 **`7da28b7dd127f002986711ea62249b73de723739`**，提交日期2026-06-03，题目 `feat(ui): move debug HUD controls to side rail`；本报告不是浮动 main/master 的摘要。
- 临时源码路径：`D:/program/sightguard-audit-roana-20261011`。GitHub网页读取超时后，使用浅层源码检出核实实际实现；未构建Android/iOS，未运行其模型或测试，未引用其性能作为本机验证。
- [ARCHITECTURE.md（固定版本）](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/ARCHITECTURE.md) 描述phone-first、原生平台运行时、共享领域契约。
- [STATUS.md（固定版本）](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/STATUS.md) 文内更新日期2026-05-31。上游称Android QNN设备短门禁及30分钟thermal gate已通过，仍待已知走廊人工陪同验证；iOS有本地/replay实现，仍待实体iPhone证明。这里是**上游记录**，不是本轮复现。
- iOS默认scheme为no-model相机路径，YOLO-only和YOLO+Depth显式启用；不能将目标系统图当成所有默认运行模式都加载模型。

源码检出纪律记录：首次clone意外触发Git LFS过滤，已中止。临时副本中确认落地了 `app/src/main/assets/yolo11n-det-int8-smart.tflite`（2,950,488 bytes）和 `samples/iphone_0530.mp4`（16,985,741 bytes）；depth文件未确认完成，`.git/lfs/`可能有对象/未完成缓存。未使用这些文件。清除操作被自动审批以“blocked by policy”拒绝，未继续删除或改用其他方式规避；残留仅位于上述D盘临时副本，未加入SightGuard模型/数据/提交。后续源码获取应预先设置 `GIT_LFS_SKIP_SMUDGE=1`，避免重复此问题。

## 2. 核实的实现路径

下列Android文件前缀为 `app/src/main/java/com/roana/app/`，iOS前缀为 `ios/Roana/Roana/`。

| 实现 | 实际源码 | 已读职责及边界 |
| --- | --- | --- |
| Frame owner/scheduling | Android MainActivity.kt:TimingAnalyzer；iOS Camera/CameraSessionController.swift、FrameInferenceCoordinator.swift | CameraX保留最新帧，finally关闭ImageProxy；iOS串行相机回调及single-flight推理，busy时丢帧 |
| Model adapters | YoloObstacleDetector.kt、DepthAnythingRunner.kt、DepthAnythingTensor.kt、InferenceBackend.kt；iOS Models/YoloObstacleDetector.swift、Depth/DepthAnythingOutputAdapter.swift | 各自拥有平台推理与输出转换；Android QNN-required，能力/代理失败抛异常，不静默CPU降级 |
| Grid fusion | CorridorGridFusion.kt；iOS Corridor/CorridorGridFusion.swift | 相对depth grid加检测框覆盖；不是语义分割、世界地图或动态跟踪 |
| Planner | CorridorPlanner.kt；iOS Corridor/CorridorPlanner.swift | 固定15×15，向前相邻三格搜索，以长度/横向净空/直行偏好比较候选 |
| Contract | CorridorContract.kt | 集中grid尺寸、门限、reason和feedback词汇，控制同概念散落 |
| State | CorridorStateMachine.kt；iOS Corridor/CorridorStateMachine.swift | 初始STOP、非STOP连续确认、STOP立即切换，保留source decision/pending状态 |
| Orchestration | CorridorPipeline.kt；iOS Corridor/CorridorPipeline.swift | fusion→planner→state→可选feedback dispatcher，支持failSafeStop |
| Feedback | FeedbackDispatcher.kt；iOS Speech/CorridorFeedbackDispatcher.swift、YoloSpeechFeedbackPolicy.swift、SpeechFeedbackDispatcher.swift | 领域状态转语义消息/utterance，平台speaker实际提交；iOS corridor控制speech时抑制YOLO说明 |
| Motion | MotionQuality.kt；iOS Motion/MotionQuality.swift、Diagnostics/VideoReplayBenchmark.swift | 分类器与日志/测试存在；未发现接入实时传感器并约束走廊决策的主调用 |
| Parity / proof | parity/corridor-core.json、app/src/test/...、ios/Roana/RoanaTests/Parity/main.swift、STATUS.md | Kotlin/Swift各自实现，使用公共fixture对齐行为；共享概念/测试而非共享二进制 |

## 3. 十个关键问题

### 摄像头帧如何流经系统

Android的 `TimingAnalyzer.analyze` 读camera timestamp、监测帧间隔，按cadence调用检测/深度；depth帧故意避开YOLO帧，用缓存的 `lastResult.bestDetection` 融合。`maybeRunLiveCorridor` 最终只传一个best detection，不应根据融合函数接受List就宣称实时使用全部目标。推理结束的release在finally。

iOS `captureOutput` 提交同一sampleBuffer给single-flight worker；runInference调用YOLO，corridor模式再执行depth，并把bestDetection映射成列表。busy时跳过提交，不排无限队列。此处默认也未提供SightGuard式多目标跟踪流。

### 视觉模型如何解耦

YOLO与depth适配器分别拥有平台模型、预处理、解析；领域内接收规范化框/DepthGrid。平台运行时与corridor词汇集中，不让UI负责规划算法。不过Android fusion overload仍接受 `YoloObstacleDetector.YoloDetection` 和 `DepthAnythingTensor.DepthMap`，并非完全中性的模型无关契约；SightGuard应参考边界而不是照搬类型耦合。

### 检测与深度如何融合

`DepthGrid.fromDepthMap` 对每帧depth做min/max归一化，各格对覆盖像素取平均。结果是相对近远强度，**不是米**。fusion筛选score≥0.35的框，覆盖格写0.96作为障碍代理；检测框整个区域被填充，不是目标真实接地足迹，也没有遮挡层次/概率融合。可参考“先统一空间、再评价候选”的结构，不能迁移这些数字当SightGuard已验证参数。

### 空间网格表达什么

固定15×15的图像相对depth/障碍强度。没有世界坐标、相机姿态、度量净空或完整occupancy probability。cell没有单独UNKNOWN/age/provenance字段。`DepthGrid`主要校验尺寸/数组长度，常量depth范围归一化为0；不能把它当完备感知有效性校验。

### Planner负责什么

从底部中央出发，每步走向上一行的中/左/右格，memoized搜索。先检测底部中央近障碍，再比较最长路径、横向净空及直行偏好；路径太短则STOP，终点偏移决定LEFT/RIGHT/STRAIGHT。这是一个小而可测试的候选空间评价器，不是完整地图导航，也不是已经实现任意通用代价函数。SightGuard Phase B只输出候选及解释，不直接给行动方向。

### 决策状态如何稳定

状态机默认三次一致非STOP候选才切换；STOP立即生效。sourceDecision与stable command分开，pendingCommand/count明确。SightGuard已有风险等级迟滞、事件确认与提醒cooldown，分别稳定分数/身份/反馈；未来corridor稳定器应独立，不重写已有生命周期，也不机械采用帧数代替毫秒时间。

### UNKNOWN如何传播

Roana当前命令enum只有LEFT/STRAIGHT/RIGHT/STOP，没有UNKNOWN命令或三态cell；失败路径常通过reason=frame_loss/low_confidence转换STOP。MotionQuality null返回label=STABLE、reason=motion_unavailable，并使trustsGuidance=true（Kotlin/Swift相同）。这不满足SightGuard未来契约的“未知不等于畅通”要求，应保留unknown reason与证据有效性，不能直接复制行为。

### 模型失败和过期结果

Android YOLO/depth异常走failSafeStop，过大camera gap也触发STOP；iOS有帧丢失和depth失败处理。但Android融合缓存bestDetection未在该调用处验证采样时间或TTL，fusion/planner签名本身没有timestamp。iOSdepth `modelMissing` 分支与其他失败分支行为不同，可能继续YOLO说明。不能声称所有平台所有失败统一STOP，也不能声称跨模型旧结果已全面受控。SightGuard Live当前明确检查capture/inference过期，Hybrid另有TTL；未来应沿用这些长处给每份证据独立expiry。

### 用户反馈是否直接耦合模型

corridor路径通过state和dispatcher再调用speaker，具有决策与播放适配边界。Android实时 `CorridorPipeline()` 未注入dispatcher，而是由 `onCorridorState` 回调到MainActivity的 `dispatchCorridorFeedback`，再通过同一个FeedbackDispatcher提交TTS（未ready时保留pending）；不能把pipeline内可选feedbackEvent当作所有实时播放的唯一来源。iOS将CorridorFeedbackDispatcher注入pipeline。上游还保留YOLO-only描述反馈，并在corridor active时做speech所有权处理。FeedbackEvent的spoken字段表示已请求speaker，不能证明实际发声完成或用户听到；SightGuard已有OpenAL backend_reported_playing，同样应记录准确的执行语义。

### Android / iOS共享什么

共享领域词汇、阈值意图和parity fixture，原生Kotlin/Swift实现，不使用KMP共同二进制。Android QNN和iOSCore ML硬件门禁不同，状态证明也不同。SightGuard目前PC Python为主，未来iOS可迁移接口与对照fixture，不能因上游有双平台就提前开发Android。

## 4. 与SightGuard的对应与取舍

| 对照类型 | Roana核实状态 | SightGuard核实状态 | 取舍 |
| --- | --- | --- | --- |
| 上游已有、本站缺少 | depth→grid→planner→stable command可调用实现 | 尚无depth、统一grid、planner；Path Relevance只有语义连通候选 | 借鉴小型planner边界；先图像候选/unknown实验，不下载depth、不输出导航 |
| 本站应保留的差异 | 审计的主路径只传bestDetection，无BoT-SORT事件生命周期 | YOLO26n+BoT-SORT多目标、RiskEvent历史、丢失恢复、AlertEvent | 保留动态跟踪/生命周期与Risk baseline |
| 功能相似、实现不同 | command多次确认/立即STOP | 分数迟滞、事件候选确认、first/escalation/cooldown | 分清稳定对象，未来组合不替换 |
| 空间证据互补 | 相对depth+box grid；未在所读主链路发现语义地面分割 | SegFormer/ADE20K地面、接地相关性Shadow，无深度 | 语义surface和occupancy分层，分别标不确定性 |
| 反馈差异 | TTS状态变化分发；haptics是后续方向 | OpenAL Soft HRTF、cue TTL/抢占、手动模拟及浏览器演示 | 保留HRTF；统一来源/播放日志，不照搬动作提示 |
| 研究基础差异 | device gate、replay、Kotlin/Swift parity | GT3.0、Annotation Studio、A/B/C、真实数据来源/冻结协议、诊断、Path A/B/C/D | 保留本站可复现评价；补空间关系独立标注 |
| 模型实验 | 大VLM不在常驻corridor路径 | Laya Structured/Vision/Hybrid独立Shadow线程/进程 | 继续冻结，不参与正式提醒/音频 |
| 硬件不适合迁移 | QNN HTP required / ANE / phone camera | Windows PC、不同推理环境、无可靠姿态/IMU | 只借接口与测试观念，不搬delegate策略/硬件门限 |

Roana控制复杂度最值得借鉴的是：小型可测试领域算法、统一词汇与reason、显式mode/模型门禁、decision/state/feedback区别、平台适配与parity证明。它也存在缓存时间和UNKNOWN表达不足；SightGuard应保留自己在时间、日志和独立评价上的优势。

## 5. 许可证边界

[LICENSE](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/LICENSE) 是AGPL v3全文；[NOTICE.md](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/NOTICE.md) 给出AGPL-3.0-or-later约定，Swift等文件有对应SPDX头，部分已读Kotlin文件未写文件头，不能凭此视为无许可约束。NOTICE的第三方段落仍含规划措辞，不能替代各实际依赖/模型许可证审查。

本轮未复制上游算法代码、fixture或资产进入SightGuard；只记录概念、路径、行为与限制，独立设计迁移方案。未来如直接复制、改写翻译或链接形成衍生实现，应先评估AGPL对应源码、告知/分发以及网络交互条款对目标分发方式的影响，并核实具体文件和模型许可。替代方案是基于自己的契约、实验与独立实现，或取得合适授权；本轮不直接复用。
