# Android 0.3 Wall Awareness 增量修复

## 范围与根因

在 `codex/android-spatial-perception-v03` 的既有 VisionStage 预览修复上追加，保留原风险引擎、YOLO、文件分析与简洁预览。主仓库未迁移、未提交、未推送。原 UI 修复 APK 保留在 `android/deliveries/visionstage-ui-fix/`；修改前 MainActivity 与 diff 另存 `android/deliveries/wall-awareness/checkpoint/`。

故障链路分层：

1. **默认没有语义推理**：`MainActivity.spatialEnabled` 原默认 false，仅手动打开空间设置才加载 SegFormer。
2. **模型有 wall 能力但没有自动事件消费者**：ADE20K 原始 0-based class 0 为 wall。`analyzeAll` 保留它，`SemanticEvidence` 可概览它，`SpatialGridFusion` 可统计墙面支持度；Grid 默认关闭且不修改 production decision。原 RiskEngine 只消费 YOLO/跟踪对象，语义仅供研究显示或用户手动请求概览，没有墙面 AlertEvent，自然不会送入语音。
3. **音频素材缺失**：原 79 条本地音频没有“前方检测到墙面。”；自由文本概览在 AVD 无中文离线 TTS 时只显示文字。不能把 generic target 音频当作墙面播报。
4. **时序问题**：旧 `SpatialDetector.RESULT_TTL_MS=500` 是旧独立分析器常量；当前 MainActivity 使用 `SpatialSessionAnalyzer` 的 3000ms 显示期限。主路径并非直接因旧 500ms 门限完全丢弃 wall。原共享辅助线程先 Depth 后语义，最少 2000ms 一次并按耗时退避，可能延迟语义反馈。

这些是代码路径确认；不据此声称已重现用户手机上的具体 Mask 或延迟。

## 新行为

`SemanticEnvironmentEngine` 是独立的纯 Kotlin 环境存在事件管理器，不引用 PC 风险模块。支持 WALL (0)、STAIRS (53/59)、STEP (121)、POLE (93)。Door/Tree 仍仅概览，不默认播报。语义默认开启，研究图层仍默认关闭，墙面当前状态放在预览下方，与最近实际播放的语音分开。

参数由带 schema_version 的 `assets/semantic_environment_config.json` 加载：

- 中央 ROI：x 0.30–0.70，y 0.20–0.85；统计真实不重叠 Mask 单元与 ROI 的相交面积，而非外接矩形面积。
- 置信度至少 0.55；ROI 覆盖至少 12%，至少 12 个支持单元。
- 至少两个不同源时刻的有效观测，跨度至少 250ms；连续性使用源视频/相机时间，非模型运行速度。
- 4s 有限丢失恢复，15s 同类别播报冷却；持续存在的同一事件不重复。
- 新鲜度保持严格 500ms（自源帧接收/解码工作开始计时），不延长旧图像有效期。首次 `onStart` 才记录已提醒。未播放的请求只能在后续新鲜证据下重试。
- 辅助线程在资源允许时最短 650ms 一次，耗时退避及原内存/温控/YOLO 慢帧保护仍有效。语义先推理并立即回调，后运行可选 Depth；不等 Depth 完成才发语义提示。
- 每个 YOLO 输入帧更新 16×16 RGB 摘要，比较源图与最近输入。画面显著变化、IMU 快转、源帧过期、停止/跳转/会话改变会拒绝或取消墙面请求。RGB 差异只是保守补充门控，不是光流或可靠相机位姿；相似画面可能无法区分，IMU 缺失仍是限制。

环境提示使用原 SpeechController 的单槽队列、音频焦点、静音、期限和停止逻辑；增加仅取消语义请求的入口，不误取消 YOLO 提醒。YOLO/系统提醒可抢占普通墙面语音。播放已开始后，普通 500ms 到期不截断半句话；画面变化、明确缺失、静音和停止仍可取消正在播放的语义提示。

四句精确中文同时支持离线动态 TTS 与本地 WAV 降级。生成方法：

```powershell
. .\android\tools\task_environment.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File android/tools/generate_local_prompts.ps1 -OnlyMissing
```

本地 Windows Microsoft Huihui Desktop 离线合成，保留原 79 个资源，新增 4 个；哈希/文本/生成器信息写入 `android/validation/local_audio_manifest.json`。应用运行无需网络。

## 接近判断边界

`approach_enabled=false`，加载配置时拒绝将其直接打开。没有加入“前方墙面可能较近，请注意。”的触发条件。现有 Ground、相对深度、Grid、IMU 尚未在真实接近序列上联合标定；单帧面积或相对深度不构成距离证据。本轮交付环境存在提示，风险 NONE 可以与墙面存在并存，不制造 HIGH、米制距离、碰撞时间或导航建议。

## 验证与复现

```powershell
. .\android\tools\task_environment.ps1
.\android\gradlew.bat -p android testDebugUnitTest assembleDebug assembleDebugAndroidTest --console=plain
# 构建结束再启动 AVD，避免与 Gradle 叠加占用。
.\android\tools\start_avd.ps1
D:\Android\Sdk\platform-tools\adb.exe install -r android/app/build/outputs/apk/debug/app-debug.apk
D:\Android\Sdk\platform-tools\adb.exe install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
D:\Android\Sdk\platform-tools\adb.exe shell am instrument -w -e class org.sightguard.app.WallAwarenessInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

真实模型测试使用已记录出处和 SHA 的 CC0 室内桌椅/墙壁照片 `v03-room.jpg`，以及室外树木照片作为墙面负例。测试读取完整真实 SegFormer 输出，保存 wall-only Mask 叠图、类别统计、ROI 支持、模型耗时及实际 MediaPlayer 回调。重复静态照片只能证明这组照片上的链路，不能代替移动相机性能、泛化准确率或真机路测。JVM 单测中的 Mask 属于明确的合成状态测试。

测试与交付实测数字以本轮交付目录中的日志、JSON 和下方最终记录为准。

## 2026-10-11 最终验收记录（Asia/Shanghai）

- JVM：98 passed，0 failed，0 skipped，包含新增 10 项语义状态/支持度/时序/音频映射测试。
- Android 16/API36/x86_64 AVD：新增 `WallAwarenessInstrumentedTest` 2 项通过（13.896s）；真实 YOLO 与风险阶段、视频解码/跳转、前台自动启动、实际语音播放、SegFormer 实际照片与同张量对照共 11 项回归通过（46.076s）。
- 真实室内照片：SegFormer 接受 2995 个语义单元，其中 1485 个 class 0 wall。提示 ROI 内达到较高置信度门限的墙面覆盖 37.53513%，面积加权置信度 0.8762927。8 次真实推理耗时依次 248、207、177、184、148、137、135、145ms。模型输出不是人工标注，也不是墙面检出率基准。
- 实际本地音频：事件 `semantic:100:0:WALL:1`，`onStart elapsedMs=125129`，`validUntilMs=125370`，`resource=environment_wall`，`detail=local_media_player`，实际文本“前方检测到墙面。”；完成于 127802ms。8 次持续墙面观测仅一次 onStart。随后对真实室外树木照片运行模型，墙面支持不足，不产生墙面提示，原事件超过恢复窗口结束。
- 主应用 CameraX：将真实照片放在 AVD virtualscene 墙面海报上，使用实际 ImageProxy 和实际 ONNX 推理，没有注入检测或语义结果。4 核/1536MB AVD 中，`semantic:1:0:WALL:1` 实际 onStart 在 36514ms，源帧接收时刻 36112ms，起播源龄 402ms，早于 36612ms 截止；完整播放完成，同一稳定事件没有重复播报。后续离开中央墙面、恢复窗口结束及重新进入产生另一个事件；静音期间不发声，解除静音后根据新鲜证据进入音频链路。
- 主应用快速转向：用 Emulator PhysicalModel 的 SMOOTH 旋转驱动实际相机与 IMU，三个实际观测分别记录源龄 283、293、493ms 且 `sceneCurrent=false, rapid=true`，支持度不被消费、不用旧墙面证据发声。停止转动后必须重新获得连续有效证据。
- “风险 NONE / 无检测目标”和墙面存在可共存，见 `camera-wall-none.png` 及 `camera-after-turn.png`。前者使用原 CC0 照片中 `[270,30,620,360]` 的真实墙面像素裁剪作为海报；裁剪来源/方法/哈希保存在 `wall_photo_crop.json`，无合成填充。后者是转向后的 AVD 虚拟室内墙壁，不冒充真实街景。
- UI 保持预览干净，未自动打开 Mask 图层；wall-only 实际模型叠图另存 `wall_mask_actual.png`。最近实际播放文本与当前状态分开。停止后图像与模型资源释放，AVD 在验收后关闭。
- 对修改的 Python 验收辅助脚本，Ruff 检查与格式检查通过；`git diff --check` 通过。没有修改 Python/PC 核心模块，因此没有重复运行无关 PC pytest。

保留的失败与边界：

- 首次设备测试在消费语义前计算研究 Grid，超过 500ms 后被拒绝，测试统计读取空支持度列表失败；已将测试按生产优先级改为先消费语义，再检查 Grid。
- 另一轮冷启动真实推理达到 653ms，正确拒绝。测试报告现独立记录 Mask 统计与事件是否接受，不把冷启动拒绝伪装成“未识别”；测试仍要求后续新鲜连续观测实际起播、仅播放一次、负例退出。
- 2 核 AVD 同时执行 CameraX/YOLO/SegFormer 时，多次源龄 0.6–0.9s，出现过请求在起播前过期；没有放宽 500ms。主界面最终验收使用相同内存的 4 核 AVD，日志同时保留 2 核失败/受限情况。AVD 性能不能替代用户手机性能。
- 用户报告 Redmi Turbo 4 Pro 实际约 23 FPS，本轮记录为用户提供的真机结果；没有据此声称新墙面链路已在该真机验收。
- 新 APK 与旧 UI checkpoint 一样，不额外捆绑约 99MB 的可选 Depth 权重；墙面存在提示无需 Depth。临时校验用的可选模型保留在交付目录，不进入 APK。接近判断仍关闭，尚无米制距离、TTC 或可靠接近趋势结论。
- STAIRS/STEP/POLE 的类别映射、独立事件及精确音频路径已实现并做合成单测；本轮真实图片/CameraX 验收重点为 WALL，未声称其他三类真实场景已完成验收。

交付目录：`android/deliveries/wall-awareness/`。可安装文件为 `SightGuard-0.3-wall-awareness-debug.apk`；机器可读汇总 `validation_summary.json`、原始模型/音频记录 `wall_awareness_actual.json`、完整 CameraX 日志 `camera-four-core.log`、构建及设备测试日志均保留。当前是未提交的独立 worktree 增量，未更改 AGENTS.md，也未迁移/覆盖主仓库工作。

最终 APK 大小 **123,954,915 bytes**；SHA-256：

`97c7ed78a15dd669549fe865523be5030a4ffcd19c825682b927b8bb70c64795`

该交付文件再次安装成功并运行实际 CameraX/SegFormer：墙面 onStart 在 95250ms，源龄 332ms，截止 95418ms；完成于 98039ms。最终安装包运行日志为 `final-apk-camera.log`，截图 `final-camera-wall.png`。对同一墙面事件，最终这轮只记录一次实际起播。最后停止感知并关闭模拟器。
