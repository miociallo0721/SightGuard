# Android 0.7 — YOLOE-26n / CameraX 替换包

2026-10-12，基于 `5a70ed3`（快速模型评测 PR #11）。用户要求制作替换 APK 并移除 ARCore。未合并 master；私人手机画面/日志、大模型、APK 均仅本地保存。

## 本次交付

- 默认使用 YOLOE-26n：原 80 COCO 提示 + 10 个桌椅、柜子、架子、柜台、显示器提示。PC 固化文本嵌入后导出 ONNX，Android 不加载文本编码器、无需联网。
- 采用官方 detection-only 架构，320 输入、0.35 阈值、端到端 `[1,300,6]` 输出。开启导出 `agnostic_nms` 的每锚点最佳类别选择，避免同一锚点同时输出桌子/茶几或公交车/卡车。三张实际图片 PyTorch/ONNX 同张量有效检测对照通过。
- 保留原 COCO ID。桌子、茶几 → 原桌子 60；凳子/扶手椅/办公椅 → 原椅子 56。新增柜子、架子、柜台、显示器使用 80..83，并保留原 `sourceClassId`。新 80..83 为概览类别，不进入既有风险资格。原风险评分核心未修改。
- 移除 ARCore Gradle 依赖、Manifest 服务查询/元数据、`ArCameraSource`、运行回调、设置开关和距离/接近/候选自动语音入口。统一使用 CameraX 的 FIT_CENTER 预览，不拉伸画面。未删除历史纯几何研究/测试代码，它们不加载 ARCore。
- CameraX 时间源为 REALTIME 时使用真实 capture timestamp 计算结果年龄；不延长 TTL。辅助模型仍采用单工作线程、资源降级及原语义门禁。
- 设置保留“使用旧版目标模型”。停止旧会话后才创建新会话；网格记录实际当前检测器 SHA。

## 实际验证

红米 Turbo 4 Pro，25053RT47C，Android 16 / API 36，所有操作经 ADB。

- **161 JVM 测试通过**；新测试验证类别兼容、近义类身份与新增类别不会获得原风险资格。
- **8 项 Android instrumentation 通过，2 项 UI 启动等待失败**：YOLOE 实际 ONNX 同张量对照、桌子图片识别、旧 YOLO 基线、视频分析/seek/异常、真实 SegFormer 墙面事件、音频 onStart/完成、语义过期取消通过。
- YOLOE 同张量框坐标最大差 `3.0517578e-5`，4 个有效公交/行人检测匹配。家具图实际输出床和原始类别 81（茶几），后者映射为桌子 60。
- `AccessibleHomeInstrumentedTest` 两项在手机 `startActivitySync` 等待超时。ActivityMonitor 方式重试亦未解决，已还原原测试，未跳过或放宽断言；具体测试框架启动原因尚未诊断。不能将此轮 UI instrumentation 说成全通过。
- **非调试 APK 已安装**，ADB 冷启动返回成功（374ms），实际 CameraX 连续输出。另用 ADB UI 树/点击验证静音/恢复、两轮停止/启动和停止状态不自行重启，均通过对应断言。
- 模型回退开关自动检查在 checked-node 断言失败；实际 release 日志随后记录旧 YOLO26n 加载、释放，再加载 YOLOE，最后观察到的是 YOLOE。不能把整段开关自动脚本说成通过。
- 实时记录到“前方有桌子”“右侧有椅子”“前方检测到墙面。”的中文 onStart。现场没有独立标注，播放链路成功不等于类别/风险判断正确。

## 真机性能与局限

短时实际 CameraX + YOLOE + 辅助模型，337 条采样帧记录：处理 FPS 中位数 **16.90**，P95 19.61；YOLO 中位 **41ms** / P95 58ms；采集到检测消费的源帧年龄中位 **194ms** / P95 224ms。采样 PSS 中位 473258KB、最大 591123KB。不是与 ARCore 的受控 A/B，不把变化全部归因于移除 ARCore。

大语义模型仍出现 5 次跨会话 256 降级，新增床/冰箱语义自动提示仍受模型门禁限制。低纹理白墙可能被语义模型分到其他类别；未降低阈值伪造墙面提醒。本轮未重新完成 TalkBack、大字体和受控移动场景完整验收。柜子/架子识别仍弱，快速评测的床类别有退步，参见 `yoloe26_quick_test.md`。

设备断开前 APK 已安装，之后停止设备操作和本任务日志进程。完整数值和限制见 `evidence/yoloe26-camerax/validation_summary.json`；原始现场日志和截图不提交。

## APK、模型与回退

APK：`android/deliveries/yoloe26-camerax/SightGuard-0.7-YOLOE-CameraX.apk`

APK SHA256：`5315f7c4aba7c344c341e11aa7976674ae1ff254bc94d06f7ec4bb95e6a01310`

版本：8 / `0.7-yoloe-camerax`，本地开发签名，365261321 bytes。包内 YOLOE ONNX SHA256：`07bb23a0d1d8ff463d19276a1a12ac007ac171487ced6dc8924e24476031ac17`。APK 检查无 ARCore 文件；Manifest 无 ARCore 元数据/查询、无 INTERNET 权限。

模型回退：设置 → 使用旧版目标模型；该开关保留 CameraX，不恢复 ARCore。旧 0.6 APK 保留于 `android/deliveries/semantic-furniture-v06/SightGuard-0.6-semantic-furniture-verified.apk`，SHA256 `635dd46dfb3048fc86ef13c612b8cf1da53193ad866b969053a3788efb20389a`。完整源码检查点 `5a70ed3`；要保留数据回退到旧实现，需要同签名且 versionCode 不低于 8 的回退构建，不能保证直接安装旧版本号成功，不自动清除数据。

## 复现

先按 `yoloe26_quick_test.md` 生成冻结提示模型及依赖。在具备 Ultralytics 8.4.174、PyTorch、ONNX/ONNX Runtime 的任务环境执行 `python android/tools/export_yoloe.py`，会生成主模型、严格清单和三张对照及测试 tensor fixtures（不提交大二进制）。官方权重来源、原始权重 SHA 和提示映射见 `app/src/main/assets/yoloe_model_manifest.json`。Ultralytics 模型遵循 AGPL-3.0 / Enterprise，未获得额外商业许可；本轮不改变此前 SegFormer 的许可限制。

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:assembleRelease --console=plain
adb -s <serial> install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s <serial> install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s <serial> shell am instrument -w -r -e class org.sightguard.app.YoloeModelInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

模型导出、Gradle 和设备验证串行推进，未启动 AVD。Python 导出脚本 Ruff 通过；失败的 UI 测试与 ADB 脚本均保留本地记录。后续优先修复新模型误检/跟踪重复提示、语义降级与设备 UI 测试启动问题，不升级未经验证的距离或导航语音。
