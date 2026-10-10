# Phase 1 架构与接口

## 模块与依赖方向

`cli → pipeline → perception / visualization`，上述模块通过 `data` 的 dataclass 交互。
`config` 管理配置；`data.replay_records` 只使用标准库，不加载视频或视觉模型。
`perception.yolo` 将 Ultralytics Results 转为 `Detection`，不向其他模块暴露张量。
`pipeline.video` 新建视频级感知器、管理资源、输出及来源记录。
`visualization.annotate` 只读取图像与帧记录，绘制 OpenCV 标注。

Phase 2 已实现：`risk` 计算空间/运动特征及评分，`decision` 管理原生状态机与提醒抑制；
`pipeline.assess` 读取既有感知 JSONL，导出风险帧、事件和提醒。
后续规划（当前未实现）：`evaluation` 独立标注与对照评估，`ui` 为 Gradio Blocks 展示。
risk/decision 不得引用 YOLO、torch、Gradio；UI 只能调用核心模块。
未来 ONNX/Core ML 可替换感知适配器；现在不实现导出或 Swift。

## JSONL 1.0

一条帧记录示例：

```json
{"frame_index":0,"timestamp_ms":0.0,"frame_width":640,"frame_height":480,"detections":[{"track_id":1,"class_id":0,"class_name":"person","confidence":0.91,"bbox_xyxy":[12.0,20.0,100.0,220.0]}],"status":"ok","error":null,"timestamp_source":"video_pts","schema_version":"1.0"}
```

帧编号从 0 开始，顺序连续。时间是视频时间，毫秒；坐标是当前帧原始像素，不是模型输入缩放尺寸。
同一帧可有多个 detection，ID 可为 null；每个视频独立运行，不把 ID 跨视频关联。
空检测为成功帧与空列表，不意味着场景安全。
处理失败为 `processing_error`、非空 `error` 与空 detections，随后终止该运行。
数据可从 JSONL 流式读取，反序列化拒绝未知版本、非法值与乱序记录。
版本不支持时显式报错，不静默套用旧字段含义。重放不依赖原始视频或权重。

## 时间与资源生命周期

输入只支持本地 MP4。OpenCV 报告的 PTS 合法且递增时采用它；否则使用 FPS 派生的单调时间，
各帧标记来源。该回退不是对可变帧率视频原时间的恢复。墙钟仅写入运行审计信息。
每个 analyze 调用新建 YoloPerceptor，模型 track 对连续帧使用 persist=True。
不得为了提速跳过空帧的跟踪更新。所有 capture/writer 在 finally 释放。
成功完成前关闭编码器，重新打开输出，检查容器帧数和首帧解码/尺寸；完整逐帧输出解码
由自动化测试及集成验证另行检查，此快速运行检查不保证每个压缩帧都无损坏。
YOLO异常写入失败帧后停止，不继续传播潜在已破坏的 tracker 状态。
提前读失败与无帧输入视为异常；没有总帧数时无法区分 EOF 与故障，显式记录这种不确定性。

## 可复现性

提交 uv.lock、配置、源码与测试；权重/输入/跟踪 YAML 哈希放入每次 metadata。
软件版本、设备、实际源尺寸/FPS、输出编码、时间来源与结束原因也保存。
模型默认参数由固定 Ultralytics 版本确定，显式传入配置中的关键推理参数。
保留原始 JSONL 用于后续风险算法对照，不重新推理生成不同感知输入。
输入、权重和环境一致仍不代表跨硬件严格逐位一致；检测/跟踪质量必须基于独立标注评估。
输出视频是恒定 FPS 的可视化副本，无音频；不是研究时间的权威来源。

## Phase 2 扩展

Phase 1 的 config/data/perception/video 接口及 schema 1.0 保留。
Phase 2 配置与结果另用 schema 2.0，详见 [phase2.md](phase2.md)。
`risk.schema` 提供明确的 dataclass 和枚举；`risk.config` 为严格的嵌套 YAML 配置。
`risk.features`、`risk.scoring`、`decision.events`、`decision.health`、`decision.alerts`
均不引用 YOLO、torch、OpenCV、NumPy 或 UI。`risk.engine.AssessmentEngine` 将这些组件串联，
每个实例对应一个视频，step 只使用已到达的当前帧和历史。
pipeline 导出的旧 analyze_video 接口改为延迟导入，保持可用，同时让 assess 完全不加载视觉库。
稳定视频标识独立于全文输入哈希；哈希仅作为审计来源，不参与事件决策。
终端 EOF 的 censored 日志属于观测终止信息，不参与或改写此前的风险分数及提醒。

## Phase 3 扩展

`evaluation.schema` 为独立真值、策略提醒及匹配记录定义 schema 3.0；保留 Phase 1
schema 1.0 和 Phase 2 schema 2.0。`dataset` 校验来源、哈希、审核状态、时间覆盖和
development/test 分组；`importing` 创建真实视频 draft 引用并在人工审核后绑定标注哈希。
`strategies` 使用同一个 Phase 2 AssessmentEngine 计算共享风险流，将 A/B 的简易决策和
C 的实际核心提醒包装为统一接口，不接收 Ground Truth。
`matching` 只评价生成后的提醒，用确定性的二分图最大匹配分配独立标注机会；
`metrics` 按合并计数及有效时长计算指标，`runner` 保存失败状态、输入证据和完整结果。
`protocol` 绑定数据、配置、源码、uv.lock 及依赖版本，test 必须校验冻结协议。
`reporting` 用原生 CSV/JSON 与 Matplotlib Agg 导出静态研究图表，不依赖 UI。

所有决策保持因果性；事后匹配可以查看完整标注和提醒集合，但不反馈到策略中。
真值事件 ID 与模型事件/track ID 分离，目标类别、人工区域、时间窗及严重程度用于匹配。
空间/尺度/运动仅为画面代理线索，不能测量真实距离、场景语义或碰撞概率。
合成轨迹用于验证实现与评价器，不是人工道路观测；当前无真实风险效果结论。

## Phase 4 扩展

`ui.app` 以 Gradio Blocks 提供四页本地控制台；`ui.services.ConsoleStore` 读取并验证已有产物，
不重算实验指标，`ui.jobs.AnalysisJobs` 为主动上传分析创建唯一新目录并调用既有流水线。
`ui.presentation`/`ui.plots` 使用原值和视频时间呈现表格与 Plotly 图，不参与决策。
新增 `analyze_video(on_progress=...)` 可选帧进度回调，旧调用接口保持兼容。
风险/决策/评价核心不引用 UI。媒体缓存和导出副本集中于 `outputs/console/`，不覆盖研究来源。
完整边界和接口见 [ui_architecture.md](ui_architecture.md)。

## Phase 5 P0–P2 实时适配

`cli → live.app → live.runtime → YoloPerceptor / AssessmentEngine`；`live.display` 只展示
实际 StepResult。采集/推理/窗口分别在采集线程、推理线程、主线程运行，单槽缓存覆盖待处理旧帧。
FrameRecord 新增 `monotonic` 时间来源，风险与事件算法保持原实现，采集序号与连续处理序号分离。
配置、过期处理、退出及验证边界见 [live.md](live.md)。保留独立 OpenCV CLI。
Gradio 的 `ui.live.LiveController` 为现有 LiveSession 添加单页面所有权、心跳和资源释放，
`ui.live_tab` 新增第五个实时页。内存 JPEG 与原始结果绑定，UI 不生成评分或提醒。


## Spatial 原生音频适配

`spatial.risk_adapter` 仅将现有 AlertEvent 转成 SpatialCue，不修改评分或事件条件。
`spatial.navigation_adapter` 提供听者相对方向的手动模拟，统一交给有界 TTL/优先级调度与
独立 AudioWorker；OpenAL Soft ctypes 后端实际查询 HRTF，失败不停止视觉流程。
产品网页通过 `ui.product_live` 复用现有控制器，研究台保留独立试听页。
摄像头先设置尺寸/FPS、最后设置 MJPG，时间戳仍使用 monotonic。
详见 [空间音频架构](spatial_audio_architecture.md)及[验证记录](spatial_audio_validation.md)。

## Hybrid Intelligence 影子研究

`live.runtime` 在原规则提醒/音频提交之后，将不可变 LiveResult 引用交给
`hybrid.manager` 的容量一最新状态槽。独立语义线程拥有选定辅助模型的子进程；
模型加载失败、超时、OOM、卸载与模式切换均不重建摄像头或 Phase 2 引擎。
`hybrid.input` 提取原图/检测/时序特征，排除规则最终风险标签；`hybrid.schema`
定义版本化结果、同源事件比较和标签融合。四种模式都是研究选择，实际 AlertEvent
及空间音频仍来自现有规则链路。产品网页通过 `/api/live/intelligence` 与已有所有权
控制器交互，研究控制台保持原入口。详见 [Hybrid 架构](hybrid_intelligence_architecture.md)。

GPU 可行性工具独立于 UI 和规则引擎。Vision 精度适配保留 FP32 readout，仅 CUDA
encoder 使用 FP16；实际参数 dtype/device 写入记录，2026-10-11 已完成真实 CUDA 前向与
固定输入实测。Windows 隔离
模型进程使用 safetensors pread 读取以规避本轮发现的 mmap 原生故障，权重哈希在加载
前验证。默认 CPU 配置保持不变。见 [GPU 研究记录](hybrid_gpu_feasibility.md)。
