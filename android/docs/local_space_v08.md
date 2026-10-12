# Android 0.8：局部空间候选与画面方向描述

2026-10-12，开发基线 `a2ecc36`（PR #12），独立分支 `codex/local-space-candidates`。本轮没有连接/操作实体手机；使用已有公开数据与一个 AVD。旧 0.7 APK 保留，不覆盖 master。

## 能做什么

前台自动感知时，复用真实 SegFormer 与同帧 YOLO，比较画面下部左、中、右的地面连通证据。连续确认后，可自动说：

- 画面左侧可见地面较连续，通行条件尚未确认。
- 画面中央可见地面较连续，通行条件尚未确认。
- 画面右侧可见地面较连续，通行条件尚未确认。

这些是**图像区域描述**。没有人体方向、净空宽度、米制尺度或路线终点，不输出前进/转向指令，不声称安全或可以通过。当前提醒与最近一次语音分开显示。用户不需要开启研究图层；研究信息放在开发者工具。

## 实现与门禁

- `LocalSpaceCandidateEvaluator` 是独立、可测试的 Kotlin 模块，复用 `SpatialGridFusion` 的 15×15 同帧证据。没有改动正式风险评分，也没有修改 PC Corridor 严格 D 组门禁。
- SegFormer 模型 SHA 必须属于已有正式契约，语义/检测来源必须匹配。网格只计入原始类别置信度至少 0.65 的语义支持。
- 从网格底边出发，检查第 8–14 行。候选格子需要至少 75% 地面覆盖，检测框占用不超过 8%，墙面/家具占用不超过 5%。未知、冲突格子不可连接；楼梯、台阶、水面、栏杆/扶手证据相交时拒绝该格子。
- 候选需有至少 65% 与底边连通的格子，各行至少两个连接格子；领先其他方向至少 15 个百分点。这里的宽度是图像格子数量，**不是人体净空**。
- 连续 3 个不同、有序的源帧确认；最长观测间隔 4 秒，消费时源帧年龄小于 1200ms。这个独立候选有效期没有修改旧墙面/风险 TTL，不需要 Depth，也没有提高辅助模型频率。
- 相机/IMU 同步未知时仅记录图像证据，不推断运动或身体方向。快速运动、受限运动、画面变化、尺寸/方向/会话改变、重复帧及过期均阻断确认。
- YOLO 主链新检测框进入已支持的地面格子，可在下一次语义推理前撤销候选。未被模型识别的障碍仍可能遗漏；这不是动作预测。
- 持续事件只在实际 onStart 后标记已播报；新事件另有 20 秒冷却。静音、暂停、停止、后台与会话切换保留现有取消行为。新的否定证据可中断当前候选语音；普通 TTL 到期不截断已经新鲜开始的完整句子。
- 候选语音优先级低于正式风险和环境结构提醒，复用 `SpeechController`。沿用已有中文离线 wav（历史文件名 `ar_corridor_*`），不含 ARCore SDK，也不增加 INTERNET 权限。

配置在 `app/src/main/assets/local_space_config.json`。诊断包含状态、方向、各区域连通率/未知率、帧键、时序支持与截止时间；物理宽度明确为 null。

## 性能修正

首次实测发现网格反复扫描全部语义单元。改为稀疏空间索引，仍使用原来的逐像素相交面积公式；新增奇数图像尺寸、随机区域与全量暴力相交对照测试。候选模块只遍历预先筛出的危险语义区域，避免反复构造类别集合。

AVD 1536MB、2 核、软件图形，单张公开图片实际 Android ONNX，18 次观测/模型：

| 输入 | 语义耗时中位数 | 网格+候选中位数/最大 | 源帧年龄中位数/最大 | 采样 PSS 中位数/最大 |
| --- | ---: | ---: | ---: | ---: |
| 256 | 105.5ms | 25.5 / 103ms | 141 / 273ms | 255090 / 263231KB |
| 384 | 233.5ms | 45.5 / 131ms | 287 / 449ms | 355632 / 487062KB |

这不是手机性能、全应用峰值内存或受控速度提升比率。低资源 AVD 的实际 CameraX 链路明显慢于独立图片推理，出现几秒的辅助帧年龄和主链负载降级；过期结果拒绝提示。没有放宽 TTL 来让实时演示成功。

## 公开数据验证

复用预先固定的 28 + 60 张 ADEChallengeData2016 **validation** 图片和语义标注，不是训练集评测，也不是路径/净空真值。逐张复核既有 SHA；没有新增模型下载、没有按预测挑选或删除失败样本。

来源：[官方 ADE20K](https://ade20k.csail.mit.edu/)、[Scene Parsing 数据](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)。图像与标注的权利不同：[官方条款](https://ade20k.csail.mit.edu/terms/)将原始图片用途限制于非商业研究/教育；软件和标注为 BSD-3。不能把原有清单中的 BSD 描述理解为所有图片均可任意商用。原图、标注与生成 APK 不提交 Git。

- PC 单线程运行真实 256 SegFormer 与已冻结 YOLOE ONNX，保存全部输出供 Android 同一候选实现回放。语义推理中位数约 85ms。
- 88 张输出中：**2 张 SUPPORTED_CANDIDATE，86 张 INSUFFICIENT_EVIDENCE**。两张为 `ADE_val_00000003`、`ADE_val_00000065`，均为中央方向。
- 两张候选在对应下部三分区内，标注地面占比达到 65% 且危险语义占比低于 1%。这只是粗略语义支持检查，**不是准确率、通行率或安全率**。14 张图片至少一个区域的标注地面占比达到 65%，但占比本身不满足连通、障碍及唯一方向门禁。
- 回放使用缓存模型输出和明确标注的合成观测间隔，只验证候选规则，不计为真实连续摄像头验证。
- Android 实际推理另外使用 6 张按标注方向差异/危险区域选择的图片：186、65、165、190、218、2。每张重新执行 YOLOE + SegFormer 三次；重复静态图片不等于行走视频。
- 256 输入：186 为左侧候选、65 为中央候选，其余 4 张不提示；两条均实际到达本地 MediaPlayer onStart，开始时间早于截止时间。
- 384 输入：只有 65 为中央候选并实际 onStart，其余 5 张弃权。保留不同尺寸/平台结果差异；没有降低阈值追求更多成功。
- PC/OpenCV 与 Android/Bitmap 对 186 的结果不同，未宣称跨预处理平台输出完全等价。两端模型 SHA 固定，完整记录保留。

## 验证等级与限制

- 合成/JVM：确认、去重、冷却、并列、断开地面、未知格子、危险语义、遮挡、源不匹配、过期、重复/倒序、会话隔离、转动和新障碍撤销。
- 真实模型：上述 PC 88 张及 AVD 6 张×两个输入尺寸。
- Android 音频：三方向固定音频的 onStart、旧候选取消、风险语音保留和实际模型派生候选播放。AVD 采用无宿主音频模式，验证 Android 播放生命周期，**没有人工听感或实体扬声器验收**。
- 原功能回归：YOLOE/YOLO 同张量验证、文件解码/seek/异常、墙面事件与实际中文 onStart、三轮相机启停释放。
- 普通界面测试首轮在有效相机帧断言失败，单独重试通过；保留失败，不改断言。没有重新执行完整 TalkBack、大字体或真实手机行走验收。

尚未证明：透明玻璃障碍、未识别障碍、地面落差、窄缝人体通过、身体朝向、行人动作/意图、接近速度与真实安全路径。近期应优先使用带运动的公开视频和独立区域标注，验证相机转动与动态障碍进入，而非扩大语音承诺。

## 复现与回退

先按 `semantic_furniture_v06.md`、`yoloe26_quick_test.md`、`yoloe_camerax_v07.md` 准备原模型与固定数据清单。使用已有含 OpenCV、NumPy、Pillow、ONNX Runtime 的环境：

```powershell
python android/tools/evaluate_local_space.py
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:assembleRelease --max-workers=1 --console=plain
adb -s emulator-5554 install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5554 install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s emulator-5554 shell am instrument -w -r -e class org.sightguard.app.LocalSpaceInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
adb -s emulator-5554 shell am instrument -w -r -e local_space_size 384 -e class org.sightguard.app.LocalSpaceInstrumentedTest#actualOnnxPhotographsAndAudio org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

旧包保留：`android/deliveries/yoloe26-camerax/SightGuard-0.7-YOLOE-CameraX.apk`，SHA256 `5315f7c4aba7c344c341e11aa7976674ae1ff254bc94d06f7ec4bb95e6a01310`。新包 versionCode 9；不保证直接安装低版本号可回退。保留数据的完整回退需从 `a2ecc36` 在独立分支构建同签名、versionCode 不低于 9 的版本。无需恢复 ARCore，不自动清除用户数据。

结构化结果与最终包哈希见 `evidence/local-space-v08/validation_summary.json`。本轮未安装到用户手机。

## 最终构建与包

168 项 JVM、64 项 Python 回归与 Ruff 通过。Android 12 个不同测试场景首轮 11 通过、1 失败，失败项单独重试通过；另做 384 输入和最终主动取消复测均通过。最终 APK 已安装到 AVD，冷启动 1198ms；实体手机未安装。Release 停止按钮的补充 ADB 检查被系统 Accessibility Suite 通知权限弹窗遮挡，未改动该权限；此前 debug 三轮启停释放测试通过。AVD 已关闭。

APK：`android/deliveries/local-space/SightGuard-0.8-local-space.apk`

SHA256：`bab43c6c4fbb119a769942b7d8e7b34ed09c4ade4568b58847d4c2d4d186712f`
