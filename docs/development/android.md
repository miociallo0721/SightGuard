# Android 开发

## 环境

使用 JDK 17 或更高版本（现有构建环境为 JDK 21）、Android SDK Platform 36，以及仓库自带的 Gradle Wrapper。Wrapper 固定 Gradle 9.1.0，Android Gradle Plugin 固定 9.0.1。

设置 `JAVA_HOME` 与 `ANDROID_HOME` 指向本机安装位置；也可以在不提交 Git 的 `android/local.properties` 中设置 `sdk.dir`。不要复制别人的绝对磁盘路径。`android/tools/task_environment.ps1` 是原开发机的 D 盘环境辅助脚本，不是通用安装器。

## 模型与构建

先按[模型清单](models.md)准备资产。ONNX 文件不会随 Git 克隆下载；模型哈希与类别契约必须匹配。编译器不负责判断资产是否齐全。

从仓库根目录运行：

```powershell
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleRelease --console=plain
```

输出为 `android/app/build/outputs/apk/release/app-release.apk`。当前 release 使用开发签名，见[发布指南](releasing.md)。Linux/macOS 使用 `./android/gradlew`；现有设备交付主要在 Windows 验证。

默认 `gradle.properties` 限制单 worker、1536MB 堆和非驻留 daemon，避免与模拟器、模型推理争抢资源。

## 设备验证

仅操作已授权的 ADB 设备；锁屏或未授权时不要绕过系统权限。

```powershell
adb devices -l
adb -s <序列号> install -r android/app/build/outputs/apk/release/app-release.apk
adb -s <序列号> shell am start -n org.sightguard.app/.MainActivity
```

设备测试包需独立构建、安装并准备相应的本地图片、视频或 tensor fixtures：

```powershell
./android/gradlew.bat -p android :app:assembleDebug :app:assembleDebugAndroidTest
adb -s <序列号> install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s <序列号> install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s <序列号> shell am instrument -w -e class <测试类全名> org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

JVM 和 Android 的规则对照都读取 `app/src/sharedTest/assets/parity_reference.json`，无需再复制第二份。其他被忽略的模型/媒体 fixtures 按对应测试与 [Android 技术索引](../../android/docs/README.md)准备，不将缺素材的运行计为通过。

## 修改边界

保持正式风险、语义环境事件、研究证据与 UI 分层。增加模型必须同时明确来源、时间戳、有效期和资源释放；隐藏图层不能成为停止正式语义提醒的隐式开关。详细边界见[架构](../architecture.md)与 [AGENTS.md](../../AGENTS.md)。
