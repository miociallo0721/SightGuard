# 空间音频验证记录

本页区分自动 mock 验证、真实设备状态验证和主观听感确认。它们验证的内容不同，不能相互替代。这里不把 mock 结果表述成硬件验证。

## 自动化与 mock

`tests/test_openal_backend.py` 使用 fake ctypes library 检验 OpenAL 函数绑定、设备/HRTF 状态处理与资源清理，不要求耳机或真实声卡。其他空间音频测试覆盖 cue、适配器和调度行为。覆盖坐标、配置、序列化、时效、去重、队列上限、抢占、暂停恢复、取消、线程所有权、音频失败降级、实时会话隔离和页面会话 API。最终完整测试结果见下文。

## Windows 真实设备

使用官方 OpenAL Soft 1.25.2 Windows x64 release 的 `soft_oal.dll`（路径：`models/openal-soft/soft_oal.dll`，Git 忽略）。该二进制来源于[官方发布包](https://github.com/kcat/openal-soft/releases/download/1.25.2/openal-soft-1.25.2-bin.zip)，许可证为 [LGPL-2.0-or-later](https://github.com/kcat/openal-soft/blob/1.25.2/README.md)。没有从上游仓库复制源码树。

真实设备后端 `open → 播放方向提示音 → 查询状态 → close` 记录中，OpenAL Soft 报告 HRTF supported、enabled，状态为 `enabled`，数据集为 `Built-In HRTF`；播放期间后端报告 source 正在播放，最终设备和上下文正常关闭。较早的原始记录 [p0_audio.json](../outputs/spatial_validation/p0_audio.json) 使用泛用设备标签 `OpenAL Soft`。随后一次只打开、查询、关闭设备的实测得到设备名 `OpenAL Soft on Speakers (Realtek(R) Audio)`，用于确认实际 endpoint。

用户佩戴耳机试听后确认能听到提示音，左右方向对应正确。前后方向及高度方向尚未验收；不能据此声称所有耳机、声卡或 HRTF 数据集都具有相同定位效果。P0 记录中的 cue 是人工构造的左、正前、右测试 cue，不是模型预测，也不验证风险方向适配的端到端准确率。

## 已知边界

设备成功打开或 `is_playing` 返回真只说明后端创建了可播放的 source，不等于用户必然听见或定位正确。听感结果依赖耳机、声卡、系统混音和个人听觉；本记录只覆盖这台 Realtek 输出设备和一次耳机左右确认。前后/高度感、不同硬件、长时间运行和风险方向映射仍需独立验证。

空间提示音属于辅助提醒。任何方向不确定、感知异常或过期情形都应保留状态与失败原因，不能将声音解释为碰撞预测、可靠避障指令或安全保证。


## 本次额外实机与构建验证（2026-10-10）

- `uv run sightguard live --camera 1 --device cpu --spatial-audio --duration 7`：
  Xiaomi 摄像头读到 640×480，最后记录 processed=5、status=ok、objects=0；
  HRTF enabled / Built-In HRTF，退出释放摄像头与线程。本场景无风险事件，不能验证风险效果。
- 合成目标框经真实 AssessmentEngine 和 RiskCueAdapter 生成 risk_initial/risk_escalation，
  真实 AudioWorker/OpenAL 均提交成功且 backend_reported_playing=true；
  日志 `outputs/spatial_validation/p1_synthetic_native.json`（忽略提交）。
  这是合成输入的端到端链路验证，不是道路场景识别或新的主观听感验收。
- `SIGHTGUARD_TEST_MODEL=models/yolo26n.pt` 下运行 `tests/test_integration.py`：2 passed，
  使用已有本地权重，无在线下载。
- `uv build`：成功生成 0.4.0 wheel 和 sdist；检查 wheel 包含 spatial worker 和网页资源。
- `uv run python scripts/prepare_openal.py`：已有 DLL 时保留并退出；下载分支尚未以该脚本重新执行，
  本机 DLL 来自同一官方发行包的此前手动下载。

不把 enqueue_to_submit_ms 解释为耳机实际发声延迟；未执行长时间稳定性、全面性能或 Android 验证。


## 摄像头画面问题与修复

此前 7 秒 CLI 验证只检查了帧读取与退出，未检查画面内容；用户随后发现黑屏和彩色横纹。
诊断中 DirectShow 默认 YUY2 的原始帧同样异常，故问题不在网页 JPEG 显示。
该机 Xiaomi 支持 MJPG；在尺寸与 FPS 设置之后最后设置 FOURCC=MJPG 才保留有效格式。
修复后运行 `uv run sightguard live --camera 1 --device cpu --spatial-audio --duration 5`，
读取正常 640×480，最后 processed=29、objects=1，现有引擎实际产生 perception_limited 与 first。
用户确认“正常了”。本次短运行日志的 FPS 不是专项性能结论；首次模型初始化曾产生过期提醒。
默认采集配置 camera_fps=30、camera_fourcc=MJPG，可在 live.yaml 调整，时间仍来自 monotonic。


## 最终检查与产品网页验证

2026-10-10 恢复任务后：`uv run pytest -q` 为 **284 passed, 2 skipped**。
两项默认跳过的真实模型集成已单独使用本地权重运行通过（2 passed）。
`uv run ruff check src scripts tests`、`uv run ruff format --check src scripts tests`、
`node --check src/sightguard/ui/web/app.js` 均通过。
采集格式测试分别检查默认尺寸与非默认尺寸的协商顺序，避免 MJPG 被尺寸设置重置。

产品网页在真实 Xiaomi 会话下展示已启用 HRTF/Built-In HRTF 和 Realtek 输出；
检查了静音/解除静音、导航暂停/提交/恢复/取消的控制与实际设备退出。
真实室内场景中可出现行人和既有引擎事件，未评价风险准确率。
页面隐藏/卸载包含主动释放请求，后端以租约超时兜底；隐藏事件尚未独立做浏览器切页验收。
网页预览在 `outputs/spatial_validation/product_page.png`（忽略提交），
原始异常帧只用于本地诊断，不提交仓库。
