# SightGuard Android

Kotlin / Jetpack Compose / CameraX / ONNX Runtime Android 1.24.3，最低 Android 8.0（API 26）。应用仅在前台感知，不申请 INTERNET 权限。

当前说明对应 Android 0.4 及其独立米制 Shadow 扩展。早期 MVP、0.2 和 0.3 文档是历史记录，不能据此判断当前默认行为。

## 用户界面与默认行为

前台自动感知，用户主动停止后不会自行重启。主界面提供状态、完整预览、当前有效提醒、最近播报，以及描述、静音、开始 / 停止和设置。

SegFormer 自动运行环境语义策略，不依赖研究图层开关；默认自动提醒 WALL、STAIRS、POLE、RAILING。STEP 自动提醒因单级台阶和路缘验证不足而关闭。深度不可用时，有充分语义证据的环境存在提醒仍可成立。

研究图层在“设置 → 开发者工具”。相对深度、Grid、IMU 和墙面接近证据不直接改变原风险评分。米制研究开关默认关闭，没有距离、接近或通行自动播报。

固定提醒使用准确匹配文本的中文离线音频降级；动态描述只接受离线中文 TTS。音频成功以播放器 / TTS 的 `onStart` 为起播依据，不能将生成事件、进入队列或显示最近文字当作已播放。

## 构建前准备

需要 JDK 17、Android SDK 36、Gradle wrapper 所需依赖和本地模型。首次准备构建环境可以联网，应用运行不依赖网络。工程中的 `tools/task_environment.ps1` 是这台 Windows 开发机的 D 盘路径设置；其他机器应在当前 shell 配置自己的 JDK / SDK / 缓存路径，不能直接照搬本机绝对路径。

大模型不提交 Git。仅克隆源码不能保证生成可运行的模型 APK。必须准备：

- `app/src/main/assets/yolo26n.onnx` 与 `model_manifest.json` 对应的 YOLO26n 契约。
- `app/src/main/assets/segformer_mit_b0_ade20k_256.onnx`，SHA256 与 `SpatialAnalyzer.MODEL_SHA` 一致；ADE20K 类别表同时校验。
- `app/src/main/assets/depth_anything_v2_small_280.onnx`，与 `depth_manifest.json` 一致。
- 本实验分支若要运行米制研究，还需 `assets/metric_research/` 中四个 ONNX 图及共享权重，与 `metric_manifest.json` 一致。关闭默认开关不等于从 APK 自动移除权重。

原始模型来源、版本、哈希和导出对照分别见[0.2 历史导出记录](../docs/android_pc_parity_delivery.md)、[0.3 空间模型](docs/spatial_perception_v03.md)和[米制研究复现](docs/metric_corridor_research.md)。已有导出入口：

```powershell
# 在仓库根目录，使用已配置导出依赖的独立 Python 环境执行
python android/tools/export_model.py
python android/tools/export_segformer.py
python android/tools/export_depth.py
```

导出依赖见 `requirements-model-export.txt`；各脚本还要求本地原权重、验证图像和相应模型适配环境。这不是自动下载或一键初始化流程。SegFormer 输出在 `android/spatial-models/`，核对数值和 SHA 后再复制到 app assets。其他模型也必须检查输出路径与对应 manifest。

导出文件哈希变化时，应先完成同输入数值验证，再显式更新契约并重新验收，不能删除 SHA 校验来绕过模型不一致。

## 本机构建与 JVM 测试

从仓库根目录 PowerShell 执行；没有设备也可以完成以下构建，不能据此声称 CameraX 或真机推理验收：

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest --console=plain
```

现有配置禁用 Gradle daemon，限制单 worker、1536MB heap，使用进程内 Kotlin 编译。构建期间不要并行启动大模型评测或额外模拟器。

- 主 APK：`app/build/outputs/apk/debug/app-debug.apk`
- 测试 APK：`app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`
- JVM 报告：`app/build/reports/tests/testDebugUnitTest/`

当前构建版本 `versionCode=5 / 0.4-metric-corridor-shadow`。它是调试研究包，不是正式签名发行版。

## ADB 设备验证

只有在设备已经正常授权、可访问时执行；不绕过锁屏或系统安装审批。本轮后续公开数据集任务没有可用手机，不启动 AVD 替代真实设备验收。

```powershell
$adb = "$env:ANDROID_HOME/platform-tools/adb.exe"
& $adb devices -l
# 将 AUTHORIZED_SERIAL 替换为 devices 中已授权的设备序列号
& $adb -s AUTHORIZED_SERIAL install -r android/app/build/outputs/apk/debug/app-debug.apk
& $adb -s AUTHORIZED_SERIAL install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
& $adb -s AUTHORIZED_SERIAL shell am instrument -w org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
& $adb -s AUTHORIZED_SERIAL shell am start -n org.sightguard.app/.MainActivity
```

instrumentation 还需要显式准备真实照片、模型张量等被 Git 忽略的测试资产；脚本见 `tools/prepare_awareness_fixtures.py`、`tools/generate_parity_reference.py`、`../scripts/prepare_android_metric_fixtures.py`。TalkBack 专项需要正常可用的系统服务，不能以普通控件树检查替代。

## 已验证与尚未验证

0.4 稳定包完成 nova12 真机模型、固定音频和短生命周期回归；实时墙面提醒有 `onStart`，用户反馈中文清楚。TalkBack / 200% 字体专项在 AVD 完成，未完成 nova12 TalkBack 专项或视障使用者体验评估。

米制扩展完成 JVM 和 PC ONNX 对照，后续 ADB 安装与前台启动成功；新增模型的 Android 推理、CameraX 时效、内存 / FPS 和生命周期仍未验收。安装成功不代表新增算法工作正常。

准确的版本、APK 哈希、安装后补充记录和回退入口见[交付说明](../docs/product/releases.md)。已知漏检、误识别、时钟与性能限制见[问题清单](../docs/product/known-issues.md)。

## 文件与诊断

文件模式通过 SAF 读取本地兼容 MP4，采用抽样分析，显示与检测相同的解码图像。时间来自请求位置估计，不声称精确解码 PTS；暂停、跳转、停止和会话切换取消旧待播提示。

开发者诊断可导出有界元数据日志，不包含摄像头图像。模型、照片、APK、截图及设备日志保存在被忽略的本地目录；对外分享前仍需检查个人路径、设备信息与敏感内容。
