# Android 0.8.1：预览局部空间候选带

基线 `ece5864`（0.8，PR #13），分支 `codex/local-space-preview-band`。本次只增加已确认局部地面候选的可视化，不改变正式风险评分、候选确认阈值或语音策略。

## 实现

- `LocalSpaceCandidate` 保留真实高置信度地面像素与已确认连通格子的交集；`LocalSpacePreview` 用稀疏格子索引裁切。像素必须来自现有 SegFormer，原始类别为 floor / road / sidewalk，置信度至少 0.65。
- 只画已有唯一方向候选，连续确认、来源、帧龄和运动门禁沿用 0.8。没有候选、平局或尚在确认时不画。
- 检测框占用非零的格子不着色。主链任何新检测框与支持格子相交时隐藏整条候选带，比语音事件的 8% 占用门禁更保守。
- 不填补 Mask 孔洞，不跨越未知/障碍区域插值，不延长到远方，不生成导航箭头。淡蓝灰半透明覆盖配合文字“空间候选 · 通行未确认”。离散像素块是有意保留的证据边界，不能为了视觉平滑补成可通行区域。
- `LocalSpaceBandOverlay` 缓存每次观测的单个绘图 Path，不增加模型、位图或推理会话。使用已有 FIT_CENTER 坐标，与检测框一致；研究快照模式不叠加实时候选。
- 源帧年龄仍必须小于 1200ms；独立协程在截止时移除覆盖层，绘制回调再次检查截止时间。快速运动、会话/尺寸变化、暂停/停止及主链失效不画。主链/运动变化沿用当前画面刷新与 300ms 状态巡检；没有声称零延迟撤销。
- 无自动屏幕阅读器公告，无新增操作按钮；既有停止、静音、设置保留。

## 限制

这是相机图像中的局部地面支持范围，不是经测量的人体可通行路径。不能保证未识别物体、透明障碍或脚下落差被发现。没有人体宽度、路线终点、世界坐标或动态绕障。相机姿态未知时不推断人体前方。低频语义或场景不满足门禁时，候选带可以完全不出现；没有放宽 TTL 或门禁来追求可见效果。

## 回退

原 APK 保留在 `android/deliveries/local-space/SightGuard-0.8-local-space.apk`，SHA256 `bab43c6c4fbb119a769942b7d8e7b34ed09c4ade4568b58847d4c2d4d186712f`。本次版本码 10，原版为 9。保留数据回退应在原源码构建同签名、版本码至少 10 的回退包；不承诺普通 `adb install -r` 可以降版本。未覆盖 master 或旧 APK。

## 验证记录

- JVM：172 项通过，0 失败/错误/跳过。其中 4 项新增测试覆盖真实像素交集/孔洞、奇数尺寸格子边界、微小遮挡、时效/会话/尺寸/运动及未确认状态门禁。原有 FIT_CENTER 横竖屏映射测试继续通过。
- Release、Debug 和 instrumentation APK 构建成功，发布 lint 门禁通过。测试清理修复后重新构建 instrumentation APK 通过。
- 实体 REDMI Turbo 4 Pro（25053RT47C）、Android 16：ADB 安装 `Success`；0.8.1 / versionCode 10；普通 Activity 冷启动 `Status: ok`，232ms。
- **两次 instrumentation 失败**：`LocalSpacePreviewInstrumentedTest.actualPublicModelBandRendersThenExpires` 在 `startActivitySync` 等待 MainActivity 45000ms 超时。没有进入公开图模型推理、截图或消失验证阶段。因此不能说真实模型/候选渲染验收通过；本轮没有新的真实模型精度或耗时结果。
- 测试在启动前将自动开始感知设为 false，原先启动语句位于 try/finally 外，超时后未恢复。已将启动放入 try 并在 finally 恢复偏好；修复编译通过，但手机已断开，未再次执行。用户已被明确告知当前手机处于停止状态，可点击开始感知，并在设置恢复自动开始。此修复仅影响测试源码，已安装 APK 无需因此更换。
- 交付前手机截图验证了主界面和停止状态；没有捕获候选带阳性截图。真实 CameraX 候选撤销、行走测试、候选显示时的 FPS/内存增量、TalkBack 与大字体专项回归本轮未完成。没有将之前 AVD 的数据冒充本轮真机结果。
- 构建单 worker / 1536MB 堆，未启动模拟器或并行模型进程。构建结束后主机可用内存约 4.5GiB（单次系统采样），不代表应用内存指标。

交付 APK：`android/deliveries/local-space/SightGuard-0.8.1-preview-band.apk`，365294361 bytes。

SHA256：`2e842ee049256e65a5dab8eaeb726f0e28fd0293bf7d5022ff6d3cc4ef8a53fb`。

机器可读证据：`android/docs/evidence/local-space-v081/validation_summary.json`。

本地日志保留在 `android/deliveries/local-space/`：`preview-final-build.log`、`preview-test-harness-build.log`、`phone-preview-test.log`、`phone-preview-retry.log`。用户手机截图/XML 仅本地保留，不上传到 Git/PR。

## 复现

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android testDebugUnitTest assembleRelease assembleDebug assembleDebugAndroidTest
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -r -e class org.sightguard.app.LocalSpacePreviewInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

公开输入沿用 0.8 的 ADE20K 验证集 `ADE_val_00000065.jpg`，模型/数据来源和许可见 `local_space_v08.md`。忽略的测试素材需要先按 0.8 的工具生成；本轮没有下载新模型/数据，也没有重新宣称旧模型评测通过。该测试设计为真实推理同一张照片三次以验证组件渲染，不等于真实移动摄像头的连续观测。

下一步应先解决该设备 instrumentation 启动问题并完成阳性显示/过期截图，再检查实时转向与障碍进入时的撤销。当前以 Draft PR 交付，保留未完成的真机验收。
