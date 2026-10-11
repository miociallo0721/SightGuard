# Android 0.3 最终验收（2026-10-11）

## 交付状态

- 可安装 APK：通过。`D:\program\SightGuard\android\deliveries\v03-final\SightGuard-0.3-spatial.apk`，222,770,674 bytes，SHA256 `69539570ae1d1ffffe74cb59f86475e80d35148628a521e59e94dc7c6cdd9d07`。v2 签名验证通过；min API26 / target36，合并清单无 INTERNET。
- Android 真实 AI：通过软件级验收。YOLO26n、Depth Anything V2 Small、SegFormer MiT-B0 ADE20K 全部由 ONNX Runtime Android 1.24.3 CPU 本地执行。模型完整 SHA/实际输入输出见 [实现说明](spatial_perception_v03.md)。没有模拟检测输出。
- 语音：固定离线中文提示音真实播放通过。平台动态中文 TTS 未验证：AVD 缺少离线中文 voice，应用显示原因并采用打包录音，不自动联网；环境概览不能用通用录音冒充动态 TTS。
- 工程：`D:\program\SightGuard\android`；隔离 Git 工作树：`D:\program\SightGuard-android-pr`，分支 `codex/android-spatial-perception-v03`。未修改 PC 风险核心、AGENTS 或冻结基准。

## 构建与自动化

- 实际执行 `assembleDebug testDebugUnitTest assembleDebugAndroidTest lintDebug`。`android/parity-validation/v03-final-build-success.log`：BUILD SUCCESSFUL in 1m 38s，80 tasks。JVM 88 tests / 0 failures / 0 errors；lint 0 errors、20 warnings。
- Android16/API36 Google Play x86_64 `medium_phone`，emulator-5554：完整 instrumentation **21 tests PASS，91.705s**，含两项真实 CameraX 冷启动设置验收。`deliveries/v03-final/instrumentation.log`。
- 静态真实三模型报告重新生成：3 tests PASS，13.894s；`model-report-regeneration.log`。设备 JSON/PNG 在 `device-reports/`，已校验 JSON 解码和 PNG 签名。
- 核心/JVM 覆盖方向、事件首次/持续/结束/恢复、等级边界与迟滞、LOW静默、MED冷却、HIGH抢占、合并、静音/排队/过期/停止、权限/重启/文件模式与后台意图。真实冻结 CameraX trace 的 primary 检测过滤前后 StepResult 完全一致。
- 初期导出模型曾因99MB readBytes超过AVD Java heap而OOM，已改流式文件加载并通过真实设备重测。新增测试中不正确的MED几何及1ms时间间隔假设已纠正；没有改变PC阈值或放宽断言精度。

## 模型与空间证据

- Depth Python Torch/ORT 同输入对照通过；Android同tensor最大绝对误差 `3.4093857e-5`。本次设备load936ms，单次288ms，warm274/282/275/272ms；真实bitmap307ms。bbox内缩区域有真实median/q10/q90/有效比例。相对逆深度大值较近，不是米制距离。
- 语义同tensor最大误差0、argmax一致率1；真实room/door/outdoor/sidewalk照片检测到 wall0/floor3/door14/tree4/sidewalk11。Android双线性与Python INTER_AREA图像预处理有差异，不能据此声称精度完全等同PC。
- 真三模型同帧Grid225格，PARTIAL、motion UNKNOWN；示例UNKNOWN43/VEGETATION2/OBJECT109/CONFLICT24/GROUND47。带模型SHA和帧来源；`production_decision_changed=false`。树冠不当作脚下占用，地面不是安全路线。
- AVD Camera2 timestampSource=0/UNKNOWN，真实SensorManager有原始采样，但不能证明精确相机/IMU同步。运动质量、Grid和Depth/语义继续是Shadow证据，未替换正式风险评分。
- 快速转动规则、无效/丢失/帧不匹配有自动化覆盖；本轮未新增物理转动/真机同步验证。

## 风险与自动启动

根因与政策见 [风险相关性说明](../../docs/android-risk-relevance.md)：Android将COCO80全部送入类别无关评分，大框/中心位置、初始无运动重新归一化和降级迟滞共同造成HIGH偏多；核心本来已有三级，界面原先只展示最高ACTIVE且固定红色。

- 等级继续来自原核心 `.35/.65`，升级margin `.02`、降级margin `.05`/hold400ms，事件确认250ms/2观察。NONE独立于VALID/LIMITED/UNKNOWN/UNAVAILABLE；UNKNOWN不显示无风险。
- 原始YOLO与全量tracking records保留；只有风险输入过滤。书、鼠标、电脑、keyboard、knife作为context保留框，不自动生成步行风险。条件家具需位置+500ms/3观察。knife保守标类别不确定；COCO无pen，未声称模型混淆已解决。
- MEDIUM首次普通提示，HIGH可越过普通冷却和抢占普通播放；持续事件保持静默。最终实际CameraX日志出现HIGH与MEDIUM事件，UI独立显示当前风险和最近播报年龄。
- 首次用途说明→系统拒绝：显示原因、同次前台不循环请求；之后清数据重新测试首次直接授权，无需点击开始即实际start generation1且仅绑定一次。`first-direct-grant.log`、`permission-denied.png`。
- 主动停止→Home→返回：stopped=true、action=null，无重启；二次冷启动自动start；正常后台stop/unbind后返回新generation恢复。`manual-stop-return.log`、`cold-and-background.log`。
- 自动启动关闭冷启动设备测试通过，偏好持久保存。文件模式/URI/位置与主动停止通过saved state保留；快速反复后台的释放等待意图有JVM回归。本轮未额外操作文件模式进程回收UI场景；真实本地MP4解码三项instrumentation和原回放core回归通过。

## 声音与性能

- 同一SpeechController实际STARTED/COMPLETED三方向播放通过；前一快照PCM采集L RMS21.559/4.268、C41.039649/41.039650、R4.323/21.850（左右约5:1）。最终APK重复controller测试通过；控制器音频实现未改变。初次零帧采集失败单独保留，重试通过。
- Android16 Spatializer available=false/enabled=false/level0。仅验证stereo panning，不是HRTF或耳机定位。停止取消当前/排队语音，风险仅在实际ON_START后提交。
- AVD真实CameraX，10s暖机+40s采样：YOLO15.08FPS/PSS168895KiB；YOLO+语义13.61FPS/273973KiB；实际双模型样本13.39FPS/427866KiB；全Grid/IMU/音频11.24FPS/436546KiB。完整原始数据与模型证据在 `performance/report.json` 和逐配置log。
- 性能脚本初版读取非checkable标签节点，导致requested-depth实际双模型、requested-both未启用空间模型；保留原始数值并标意图不符，已修脚本。未把这两组按错误标签报告；**单独Depth相机对照尚缺，五配置矩阵不称为全部通过**。按用户收尾要求未扩充重测。
- 用户提供的旧版红米Turbo4Pro/Android16/语音开启约23FPS仅作为历史基线；本轮没有实体手机，不推定新版手机FPS、TTS语音包或长期稳定性。

## 截图、快照与限制

最终 `deliveries/v03-final/`：`camera-final.png`、`depth-overlay.png`、`grid-overlay.png`、`first-permission-purpose.png`、`permission-denied.png`、`manual-stop-return.png`、`cold-and-background.png`；设备真实语义四图和depth热图在 `device-reports/`。

阶段APK保留于v03-depth/v03-semantics/v03-grid/v03-spatial-pre-fixes；Depth修复版与Semantics版是同一构建快照，不能当成独立版本。initial-oom不是可用交付。0.2原APK SHA仍为 `c95b69beb9fe0c7720e189cb34bc8aec4254c059a3b26d6eefe5873289c99d25`。

还未验证：实体手机/鸿蒙兼容、动态离线中文TTS、HRTF/耳机定位、精确IMU同步、长时运行/独立行走风险校准、完整五配置性能矩阵、完整PC研究功能等价性。Depth/语义降低分辨率、类别可信度只作轻量时序处理。SegFormer权重再分发许可未独立确认。所有中间产物在D；AVD已停止以释放主机内存。
