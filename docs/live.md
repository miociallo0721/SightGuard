# Phase 5 P0–P2：PC 实时摄像头原型

本次仅完成 Git 基线、摄像头流水线和 OpenCV 风险可视化，不是 v0.5.0 正式发布或完整 Phase 5 验收。
复用 YOLO26n/BoT-SORT、AssessmentEngine、EventManager 和 AlertScheduler；未更改风险评分、
事件确认、升级及重复抑制算法。AGENTS.md 未修改。

## 启动

在项目根目录安装既有锁定依赖，准备本地 `models/yolo26n.pt`。不会自动下载权重。

```powershell
uv sync --locked
uv run sightguard live --camera 0 --device auto
# 本机 Xiaomi 摄像头的 Windows DirectShow 编号为 1：
uv run sightguard live --camera 1 --device auto --width 640 --height 480 --imgsz 640
# 可选：8 秒后自动退出，便于短时验证
uv run sightguard live --camera 1 --device cpu --duration 8
```

Q 退出，H 显示/隐藏详细事件信息；关闭窗口或 Ctrl+C 同样执行资源释放。
OpenCV 需要图形桌面及摄像头权限。仅支持本机编号摄像头，不提供 Gradio 实时流。

`--device auto` 使用可用 CUDA 的编号 0，否则使用 CPU。也支持 `cpu`、`cuda`、`cuda:1`、
现有 Ultralytics 设备字符串。显式使用 `--config` 而未指定 `--device` 时保留配置设备；
`--model` 和 `--imgsz` 覆盖感知配置。`--risk-config configs/risk.yaml` 读取既有风险配置。
`--live-config configs/live.yaml` 配置摄像头尺寸、过期时间、提示显示时间及线程退出等待上限。
命令行摄像头/尺寸参数优先于 live YAML。尺寸是驱动协商请求，以实际读到的画面为准。

Windows 使用 DirectShow 编号；本机设备列表依次为 Camo、XiaoMi USB 2.0 Webcam、
VTubeStudioCam、OBS Virtual Camera。编号随设备安装/卸载可能变化，其他机器须重新核对，
不能把本机的编号 1 当作 Xiaomi 的通用编号。Linux/macOS 使用 OpenCV 默认后端。

## 实时数据与线程

`live/runtime.py` 维护一个会话：采集线程持续读画面，容量为 1 的 LatestFrame 覆盖旧的待处理帧；
单一推理线程创建一次 YoloPerceptor，依次调用持续跟踪与既有 AssessmentEngine；
主线程仅显示结果和处理键盘。每次调用 live 创建独立模型、跟踪器、事件引擎与唯一 stream_id。
LiveSession 不允许重复启动。BoT-SORT ID 仍仅表示会话内估计关联，不是真实稳定身份。

时间为 `monotonic()` 相对于会话开始的毫秒值，以读帧结束时刻采样；不是固定 FPS 推算，
也不是摄像头传感器曝光时间。FrameRecord 的 timestamp_source 新增 `monotonic`，
既有 schema 1.0 字段与离线 PTS/FPS 来源不变。风险引擎处理编号连续；另有 capture_index
表示实际采集序号，跳号表示采集帧被最新帧机制覆盖，不向风险引擎伪造这些帧的检测。

LiveResult 绑定原始 CapturedFrame、FrameRecord 与 StepResult。窗口展示这张匹配图像，
不把旧框贴到最新采集画面；显示采集序号、时间和帧龄。超过 stale_after_ms 时隐藏框、
显示 UNKNOWN / STALE。超时推理结果被转换为 processing_error，丢弃其检测。
无新帧、摄像头失败或推理失败时，以当前会话时间向既有引擎提交失败状态心跳；
感知受限提醒仍须满足现有引擎确认时间，不能把“无有效感知”当成安全或空检测场景。
心跳节流为 100ms，仅用于提交健康状态，不改变风险阈值或提醒周期。

推理异常后不继续使用潜在损坏的跟踪器；摄像头断开与模型失败显示错误后退出。
正常退出停止线程、释放 VideoCapture、完成引擎的观测终止并关闭窗口。
对于底层驱动/原生推理调用永久阻塞，Python 无法强制取消；超过配置等待上限会显式报错，
线程设置为 daemon 以免进程永久无法退出。当前没有验证所有驱动的阻塞恢复。

## 画面内容

显示前方关注梯形、检测框/类别/track_id、实际 event_id、评分、等级及事件状态；
LOW 绿色、MEDIUM 黄色、HIGH 红色、UNKNOWN 灰色。侧栏展示实际事件快照。
没有检测目标不等于环境安全。过期数据的历史事件不显示为当前有效风险。

最近一次真实 AlertEvent 的类型、原因和时间持续可查，短时红色边框表示新提醒；
FIRST ALERT、RISK ESCALATION、PERCEPTION LIMITED 分别对应首次、升级及感知受限。
默认同级持续事件不重复提醒，可观察最近提醒列表。提示条到期不删除核心事件。
FPS 为最近最多 20 次成功感知/风险更新完成时间的实际速率，非摄像头标称 FPS；
失败心跳不计入 FPS。显示单帧检测/跟踪/风险处理耗时，尚未形成性能基准结论。
Demo 不保存私人摄像头视频、不播放音频，也不提供距离测量或可靠碰撞预测。

## 本次验证记录

- Git 基线提交 `73313da`：`chore: establish SightGuard v0.4.0 baseline`，标签 `v0.4.0`。
  暂存审查为 102 个必要文件，常见凭据模式无匹配；虚拟环境、权重、视频和输出均忽略。
  提交身份取自已连接 GitHub 账号，仅配置本仓库；无远程仓库、无 push。
- `uv run sightguard live --help`：成功，原有命令保留。
- `uv run pytest tests/test_live.py -q`：11 passed。覆盖最新帧替换、帧/结果绑定、单调时间
  与连续处理编号、空目标、缺失 ID、会话隔离、摄像头失败/线程退出、感知过期、真实引擎
  首次/升级/受限提醒、重复抑制、绘制原值及 Q/H 交互。使用 MockCamera，无联网或 GPU。
- `uv run ruff check .`、`uv run ruff format --check .`：通过。
- `uv run pytest -q`：186 passed、2 skipped；两项为需显式本地权重的集成验证。
- 设置 `SIGHTGUARD_TEST_MODEL=models/yolo26n.pt` 后 `uv run pytest -q`：188 passed。
- 真实摄像头：运行 `uv run sightguard live --camera 1 --device auto --width 640 --height 480
  --imgsz 640 --duration 8`，CPU、XiaoMi USB 2.0 Webcam、实际首帧 640×480，退出码 0。
  OpenCV 窗口和实际模型/风险调用运行成功，末次状态 ok；本次末帧没有检测目标。
  启动等待期间由引擎实际生成一次 perception_limited 提醒，正常退出释放资源。
  这次短时读取不能证明真实人员风险识别或道路安全性，未对真实目标首次/升级开展人工验收。
- 无效摄像头编号 999：日志说明无法打开，释放资源后退出码 1。
- 合成升级可视化输出在 `outputs/live_validation/mock_escalation.png`，已查看布局；
  使用真实引擎评分/提醒，但输入框和耗时/FPS 是合成展示参数，不是摄像头性能数据。

未执行 CUDA 验证、长时间稳定性或全驱动兼容测试。真实场景检测/风险质量仍需独立标注验证。
本次到 P0–P2 停止；不创建 v0.5.0 标签，不新增音频、移动端或模型优化功能。
