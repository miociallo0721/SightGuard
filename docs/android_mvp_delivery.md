# SightGuard Android MVP 交付记录

日期：2026-10-11。研究演示原型，辅助盲杖与导盲犬，不提供可靠碰撞预测或安全导航保证。

## 三项交付状态

- **APK 构建交付：通过。** 已实际运行 `assembleDebug`，APK 存在、签名校验通过，已安装并启动 AVD。
- **Android AI 推理：通过软件级验证。** APK 内 YOLO26n 在 Android ONNX Runtime 上真实执行；真实照片与 Python 对照通过；CameraX 连续帧已显示行人、公交车中文框并触发事件。AVD 的照片海报输入不是实际街景录像或真机测试。
- **平台中文 TTS 播放：未验证。** 当前 AVD 没有可用的离线中文 voice，应用明确显示原因。**固定本地中文提示音播放通过**：实际 MediaPlayer `start`/`isPlaying` 后记录 `SG_LOCAL_AUDIO onStart`，包括开启、停止与风险提醒。固定录音不等于完整 TTS；应用不自动联网找语音包。

## 交付文件

- APK：`D:\program\SightGuard\android\app\build\outputs\apk\debug\app-debug.apk`
- APK 大小：102,014,629 bytes；SHA256：`ff9c225798856555173a4b3de4916e71868636526b0ecd6e4f01197c7f9e6469`
- 工程：`D:\program\SightGuard\android`
- 最终构建日志：`android/final-build.log`，`BUILD SUCCESSFUL in 59s`，80 tasks，16 executed，64 up-to-date。
- 截图：`android/validation/sightguard-detection-front.png`（实时真实模型输出的中文框）、`sightguard-restarted.png`（默认房间预览）。
- 单图验证：`android/validation/android_model_validation.json`、`python_model_validation.json`、`android-instrumentation.log`。
- 最终相机/语音验证：`android/validation/final-camera-events.log`、`final-muted-events.log`、`session-events.log`、`camera-background.txt`。修复前记录另存，不能当作最终防重复通过证据。
- APK 签名和权限：`android/validation/apk-signature.txt`、`apk-info.txt`。CAMERA 权限；无 INTERNET 权限。

APK 支持 Android 8/API 26 以上，ABI 为 arm64-v8a、armeabi-v7a、x86_64。debug 签名，仅用于研究演示。运行所需模型和固定中文提示音均在 APK 内，不需要 PC、Python 或云端服务。

## 真实模型与输入输出

- 原始权重：`models/yolo26n.pt`，SHA256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
- APK 模型：`android/app/src/main/assets/yolo26n.onnx`，9,767,944 bytes，SHA256 `e093d7e1805d63cc1b671c333ce0b48b5cd33f70f64ea16c9b5bd713e33242cf`。
- 导出：Ultralytics 8.4.174、PyTorch 2.14.1+cpu，ONNX opset 17，固定 320，end-to-end，dynamic/simplify/nms 均 false。详细清单在 `model_manifest.json`。
- 唯一推理运行时：ONNX Runtime Android 1.24.3 CPU；Python 对照 ONNX Runtime 1.24.4。
- 实际输入 `images float32 [1,3,320,320]`，NCHW RGB/255，居中 letterbox，114 填充。
- 实际输出 `output0 float32 [1,300,6]`，每行 `[x1,y1,x2,y2,score,class_id]`；该实际 end-to-end 导出已选择 Top-K，不再套用其他 YOLO 版本格式或额外 NMS。
- 真实照片为 Ultralytics bundled bus.jpg，810×1080，SHA256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`。
- 相同 320 输入与匹配的双线性预处理下，Android 与 Python 均检出 5 个目标（4 行人、1 公交车）；bbox IoU 0.99397–0.99898，最大置信度差 0.01150；最终 APK 的 Android 冷加载加单图推理 200 ms。这里只验证此照片，不能据此声称视频准确率或泛化效果一致。

## 已运行验证

- AVD：`medium_phone`，Google `sdk_gphone64_x86_64`，Android 16/API 36，Google Play x86_64。无物理 Android 设备。
- CameraX：后置 virtualscene，真实 RGBA ImageProxy、row/pixel stride、crop 和 rotation；实时分析 upright 480×640，传入固定 320 模型；KEEP_ONLY_LATEST、单线程推理和模型串行关闭。
- 默认虚拟客厅持续真实推理，部分视角 0 目标；变化视角也出现书架被误判为椅子等误检，未伪造或屏蔽输出。将真实 bus.jpg 放入 AVD wall 海报并移动模拟相机后，检测到行人、公交车，中文框与照片中目标对应。实时约 8–17 FPS、常见总分析耗时约 50–160 ms，随宿主负载变化；不是手机性能数据。
- 自动链路：真实相机帧 → YOLO26n → 连续观测风险策略 → `SG_ALERT` → `SG_LOCAL_AUDIO onStart`，已触发行人、公交车等提醒。固定提示音不覆盖的 COCO 类别明确退化为方向加“目标”。
- 最终 APK 同一海报画面实测约 46 秒，仅确认 1 个公交车与 3 个行人提醒；首轮约 11 秒结束后持续安静。原照片第四位边缘行人在该裁切摄像头画面中未形成有效检测，不能把这次实时测试描述成检出全部目标。静音新会话约 58 秒，检测持续但无风险语音；解除静音后待提醒事件正常播出，随后切后台取消音频并释放相机。
- 权限拒绝后不启动相机；再次开始授权成功。开始、停止、会话重启已实测。静音时实时检测继续。
- 切后台后相机解绑、模型释放；camera service `Active Camera Clients: []`。返回页面显示“感知已停止”、0 FPS、“摄像头已释放”，不自动恢复。
- Kotlin 单元测试 19 项通过（`android/final-replay-tests.log`）：方向、首次提醒/持续静默、事件消失再出现、事件仲裁/重要升级、抖动、过期、会话重启、语音初始化失败、静音、停止、迟到回调等；包含实际 CameraX 检测记录中公交车重复框及完整 200 帧回放，完整记录只提醒 1 bus + 3 person。
- Android instrumentation 2 项通过：真实 ONNX 照片对照，以及 RGBA crop/stride/0/90/180/270 旋转。
- `lintDebug` 0 errors、15 warnings（版本更新建议等）；没有删除或弱化测试。Python 回归 `567 passed, 27 skipped`，限制 OpenBLAS/OMP 为单线程以适配本机内存；Ruff 通过。原 Python 源码、AGENTS.md、冻结 Benchmark 未因 Android MVP 修改；没有 push 或合并。

## 与 PC 版的实际区别及限制

PC 默认为 640、confidence 0.25、类别过滤 `[0,1,3]` 和 BoT-SORT；Android 为 320、confidence 0.35、COCO-80 类别与简单图像框匹配事件。Android 小目标和弱置信目标可能更容易漏检，类别范围不同；没有迁移完整 PC 风险引擎、深度、走廊、运动估计或跟踪器，不能宣称策略跨语言等价。`pc_default_comparison.json` 是额外的一张照片对照，不是视频 benchmark。

Android 事件阈值集中在 `RiskPolicy` 常量；两次连续观测确认、2.5 s 消失确认、10 s 内强几何证据的近期事件复关联、3 s 普通提醒冷却、方向滞回、重要图像目标可抢占。同帧同类别且中心距≤0.03、IoU≥0.80 的框只创建一项提醒，不删除界面上的原始模型检测。该修复来自实际 CameraX 的 200 帧记录，解决同帧重复公交车框创建重复事件；记录保存于 `android/validation/camera_detection_trace.jsonl`（SHA256 `be49f2daffbd9f04ad56643d1aa7373283e5135eca6352146cabe5fb4ed60e0a`）。大小/中心仅是图像代理，不是距离、碰撞概率或接近证据；不播报“前方安全”。结果 1.5 s 过期清框，长时间无有效结果才提示中断。

尚待实测：真实街景视频的漏检/误检和重复率、手机 CPU 性能/发热/长期运行、离线中文 TTS voice 的实际平台播报、无摄像头/相机占用/模型损坏的设备级错误路径、鸿蒙 APK 兼容层与 ADB。仅前台运行，没有视频文件导入、后台服务或完整导航功能。音频焦点丢失或生成提醒与静音切换同时发生时，当前提醒可能被取消。

## 复现与资源约束

在仓库 PowerShell 中先执行任务局部环境脚本，再构建：

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android assembleDebug testDebugUnitTest lintDebug assembleDebugAndroidTest --console=plain
& "$env:ANDROID_HOME/platform-tools/adb.exe" -s emulator-5554 install -r android/app/build/outputs/apk/debug/app-debug.apk
& "$env:ANDROID_HOME/platform-tools/adb.exe" -s emulator-5554 shell am start -n org.sightguard.app/.MainActivity
```

单图 instrumentation 需要先安装 `app-debug-androidTest.apk`，再执行 `adb shell am instrument -w -r org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner`。测试照片未打入正式 APK；由 `tools/export_model.py` 导出/验证时生成测试 assets。模型不提交 Git，导出依赖记录在 `requirements-model-export.txt`；导出专用环境需使现有 `.venv/Lib/site-packages` 可见，例如设置任务局部 PYTHONPATH 后运行 `.export-venv/Scripts/python.exe tools/export_model.py --validate-only`。

已有本次导出环境时，从仓库根目录复核模型的准确命令：

```powershell
. ./android/tools/task_environment.ps1
$env:PYTHONPATH = (Resolve-Path .venv/Lib/site-packages).Path
& android/.export-venv/Scripts/python.exe android/tools/export_model.py --validate-only
```

`tools/start_avd.ps1` 启动本次低占用 AVD。用户主动开始感知后，用安装 grpcio/grpcio-tools 的导出环境执行 `tools/avd_camera_fixture.py`，将实际照片贴到 wall，再把虚拟相机移动至正视海报位置。它只修改模拟器输入，不向 APK 注入检测输出。debug APK 前 200 个有效帧保存真实检测 trace 和首帧 RGB 到应用私有目录，供本地验收；release 构建禁用这项记录。

SDK、Gradle cache、UV cache、构建、下载、生成 proto 和任务临时文件置于 D。AVD 数据盘为 `D:\AndroidAvd\medium_phone.avd`。Gradle 1.5 GB heap、1 worker、Kotlin in-process、无常驻 daemon；AVD 2 cores、540×1200、host GPU、无窗口/快照。API 36 镜像将请求的 1536 MB RAM 自动提高到 2560 MB，不能声称只用 1.5 GB。C 盘恢复约 12 GB 可用；约 603 MB Blender 占用的旧 UV 文件按用户要求保留，已备份到 D，未终止 Blender。

验收结束已关闭 AVD，未留下 Gradle/Kotlin 编译 daemon。模拟器内部默认写入的 C 盘 `Temp\avd` 控制元数据目录也在模拟器关闭后迁移至 `D:\Temp\SightGuard\emulator-control`，原路径为 junction。
> 后续0.2原生PC体验对齐版（本地视频、风险核心、可选真实SegFormer）的最新交付见 [android_pc_parity_delivery.md](android_pc_parity_delivery.md)。本文保留0.1 MVP原始验收记录及独立APK副本。

## 2026-10-11 Android 0.3 更新

最终 APK：`D:\program\SightGuard\android\deliveries\v03-final\SightGuard-0.3-spatial.apk`。
SHA256：`69539570ae1d1ffffe74cb59f86475e80d35148628a521e59e94dc7c6cdd9d07`。
88项JVM、21项AVD instrumentation通过；真实三模型及固定中文音频通过，平台动态中文TTS尚未验证。
完整验收与限制见 `android/docs/spatial_perception_validation.md`；0.1历史APK保存在 `android/deliveries/mvp-verified/SightGuard-0.1-mvp-verified.apk`，0.2保存在 `android/deliveries/parity-final/SightGuard-0.2-parity.apk`。
历史正文中的build输出路径会随新构建变化，不能拿当前app-debug.apk对应旧SHA。
