# SightGuard Android

原生 Kotlin / Compose / CameraX / ONNX Runtime Android 1.24.3，Android 8.0（API 26）以上，前台运行。YOLO 主链和 SegFormer 环境语义按系统策略运行，用户无需开启研究图层。应用无 INTERNET 权限。

当前 **0.6 家具语义增量**：[实现与验证](docs/semantic_furniture_v06.md)。新增 8 个家具组的概览、连通区域证据、网格占用与独立语音资源。自动播报仅按实际模型与类别验证结果开放；正式 YOLO 风险评分不变。384 输入推理不及时或主链负载过高时，单会话退回原 256 模型，新增床/冰箱自动提醒随之关闭。

0.6 构建需要额外的 `app/src/main/assets/segformer_mit_b0_ade20k_384.onnx`。它使用原冻结权重、同一预处理和单独 SHA 契约；导出及验证命令见 0.6 文档。两个模型不会同时持有 ONNX Session。缺少较大模型时可以降级，但这样的构建不能声称包含已验证的新家具自动提醒。

本分支新增 **0.5 ARCore 实验接入**：见 [红米实测、失败记录及回退说明](docs/arcore_awareness_v05.md)。可用设备启用 ARCore 相机和空间证据；不支持或关闭实验开关时使用 CameraX。墙面距离、相机接近与地面候选尚未通过受控实景验收，不提供安全路径或数值距离播报。不要将新增语音素材播放成功等同实际测距成功。

应用默认在前台自动感知，保留显式停止与静音；离线中文 TTS 不可用时，固定环境提示有匹配的本地音频。0.4 功能记录见 [Accessible Awareness](docs/accessible_awareness_v04.md)。早期 MVP 验证见 `../docs/android_pc_parity_delivery.md`，原副本保留在 `deliveries/mvp-verified/`。

ARCore 非调试实验包使用 `:app:assembleRelease` 构建，仍为本地开发签名。真机性能与调试包明显不同；对照条件、源帧年龄和局限见上述报告。旧 APK 不覆盖，设置中关闭实验性空间提醒即可回到基础感知。

## 构建与设备测试

在仓库根目录 PowerShell 执行。先关闭 AVD 再构建，以降低内存占用。任务缓存、SDK、AVD、临时文件均使用 D 盘。

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest --console=plain
./android/tools/start_avd.ps1
$adb = "$env:ANDROID_HOME/platform-tools/adb.exe"
& $adb wait-for-device
# 等待 adb shell getprop sys.boot_completed 返回 1，再安装。
& $adb install -r android/app/build/outputs/apk/debug/app-debug.apk
& $adb install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
& $adb shell am instrument -w org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
& $adb shell am start -n org.sightguard.app/.MainActivity
```

默认 build 需要 `app/src/main/assets/yolo26n.onnx`、`segformer_mit_b0_ade20k_256.onnx`，模型和验证二进制不提交 Git。导出方法及实际 SHA 见交付说明。共享跨语言回放测试用 `src/sharedTest/java`，同时在 JVM 和 Android 上执行；真实模型测试独立放在 androidTest。

首次克隆须准备原始 `models/yolo26n.pt` 和 `models/path_relevance/segformer_mit_b0_ade20k_160k.pth`（来源及SHA见交付文档），在独立Python环境安装导出依赖，然后执行下面的不带 `--validate-only` 的首次导出命令。模型未准备时，编译成功不代表APK可以运行检测。

重新导出会生成新的YOLO清单。SegFormer加载时还校验`SpatialDetector.MODEL_SHA`；如导出环境导致SHA变化，须先完成脚本的相同张量数值验证，再明确更新该固定SHA并重跑设备测试，不能删除校验。交付文档中的APK SHA只对应已验收副本。

文件模式使用 SAF 选择兼容 MP4，本地解码和 CameraX 共用 VisionStage。播放为抽样分析（最高约 10 FPS），显示实际参与检测的同一张解码图像；时间来自请求位置估计，不声称解码器提供了逐帧 PTS。暂停、跳转、结束和停止会取消待播语音；跳转重建跟踪与事件状态。

## 本机复现模型和 CameraX 输入

导出环境依赖记录在 `requirements-model-export.txt`。本机 `.export-venv` 通过任务局部 PYTHONPATH 复用现有 PC 包，未改变 PC uv.lock。

```powershell
. ./android/tools/task_environment.ps1
$env:PYTHONPATH = (Resolve-Path .venv/Lib/site-packages).Path
& android/.export-venv/Scripts/python.exe android/tools/export_model.py
& android/.export-venv/Scripts/python.exe android/tools/export_segformer.py
Copy-Item android/spatial-models/segformer_mit_b0_ade20k_256.onnx android/app/src/main/assets/
Copy-Item android/segformer-validation/segformer_mit_b0_ade20k_256.tensor.bin android/app/src/androidTest/assets/segformer.tensor.bin
Copy-Item android/segformer-validation/segformer_mit_b0_ade20k_256.logits.bin android/app/src/androidTest/assets/segformer.logits.bin
& .venv/Scripts/python.exe android/tools/generate_parity_reference.py
Copy-Item android/app/src/test/resources/parity_reference.json android/app/src/androidTest/assets/
& .venv/Scripts/python.exe scripts/generate_android_video_fixtures.py
# 启动 AVD、用户点击开始感知后：
& android/.export-venv/Scripts/python.exe android/tools/avd_camera_fixture.py
# 可选真实虚拟相机空房间画面，用于无目标静默测试：
& android/.export-venv/Scripts/python.exe android/tools/avd_camera_fixture.py --empty-view
```

AVD 输入脚本只改变模拟器真实虚拟场景，不注入检测结果。照片和视频测试资产的生成、复制位置见导出脚本、`scripts/generate_android_video_fixtures.py` 和交付说明。真实街景视频带独立来源与许可记录，不打入 APK。

真实街景开发验证素材可手动运行 `android/tools/download_traffic_video.py` 获取；它保留作者、许可、裁剪方式和哈希。生成的下载、模型、截图、日志和实验输出不提交Git。交付文档引用的本机验收路径需要本地生成或从此次交付目录读取。
