# SightGuard Android

原生 Android 环境感知应用，基于 Kotlin、Jetpack Compose、CameraX 和 ONNX Runtime。运行需要 Android 8.0（API 26）或更高版本，不依赖 ARCore。

[安装与使用](../docs/guides/getting-started.md) · [构建指南](../docs/development/android.md) · [模型准备](../docs/development/models.md) · [版本说明](../CHANGELOG.md)

## 用户操作

打开应用并授予摄像头权限后，默认在前台开始感知。主界面提供“描述周围”“静音提醒”“停止感知”和“设置”。研究图层默认隐藏，隐藏图层不会关闭环境模型。

已主动停止或关闭自动开始时，可点击“开始感知”；后台不持续使用摄像头。动态中文 TTS 不可用时，固定提醒使用对应的本地音频，完整环境描述不会用不匹配的音频代替。

## 构建

先配置 JDK、Android SDK 并按[模型准备](../docs/development/models.md)放置对应的 ONNX 资产。仓库不包含权重，编译成功不等于模型已经就绪。

在仓库根目录执行：

```powershell
./android/gradlew.bat -p android :app:assembleRelease
adb install -r android/app/build/outputs/apk/release/app-release.apk
adb shell am start -n org.sightguard.app/.MainActivity
```

当前 release 配置使用开发签名，适用于现有评估交付；公开发行签名与商店发布尚未配置。详见[发布检查](../docs/development/releasing.md)。

## 目录

- `app/src/main/`：应用、配置、模型清单与本地音频。
- `app/src/test/`：JVM 测试。
- `app/src/androidTest/`：设备和真实模型测试。
- `app/src/sharedTest/`：JVM 与 Android 共享测试及唯一的对照数据。
- `tools/`：模型导出、数据准备与设备工具。
- `docs/`：各版本的技术说明和证据索引。
- `deliveries/`：本机生成的 APK、日志与回退资料，不提交 Git。

当前功能边界见[能力说明](../docs/capabilities.md)，历史技术资料见 [Android 文档索引](docs/README.md)。
