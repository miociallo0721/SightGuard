# Android 0.5 ARCore 实验接入与红米验证

2026-10-12。**部分验证的实验版本，不是已通过实测标定的测距或导航产品。**

## 交付结论

- 已接入 Google ARCore 1.56.0，REDMI Turbo 4 Pro（25053RT47C，Android 16 / API 36）实际运行，曾得到 TRACKING 和非空稀疏深度。
- 原 YOLO 风险评分不变；墙面存在、楼梯等正式环境提醒仍走原调度。实时相机记录到中文“前方检测到墙面”的真实 `onStart`。
- 新增独立的墙面平面距离估计、相机接近候选、地面连续区域候选，以及五条中文离线语音。用户本轮明确要求开启实验性提醒，因此本实验版本开关默认开启；每次播报仍须通过全部证据与时效门禁。
- **没有完成卷尺距离标定、受控接近/纯旋转正负例、人体净空或通行验证。**最终非调试版采样窗口没有足够墙面深度支持：距离、较近、接近、空间候选输出均为 0。不能把音频资源播放成功称为这些真实场景能力验收成功。
- 不播报数值距离、不声称身体正在向墙移动、不下达前进/转向指令、不输出 SAFE/PASSABLE。设置中可关闭实验性空间提醒，恢复 CameraX 基础感知。

## 代码与数据链路

`ArCameraSource` 独占相机，使用 GLSurfaceView 显示 ARCore 纹理；CameraX 与 AR Session 二选一。CPU 相机帧最多一张在途。真实 YOLO 与 SegFormer 同帧并行、各自单 worker，无辅助队列；几何消费必须等到该帧 YOLO 检测结果可用。AR 模式跳过神经网络相对/米制深度，避免重复重模型。

`ArSpatialEvaluator` 在 UI/风险引擎之外消费明确的 `SpatialFrameKey`、真实语义格、稀疏深度置信度、相机内参与 AR 世界位姿：

1. 墙面中央区域匹配 ADE20K wall=0，排除同帧 YOLO 框，要求至少 30 个高置信度点及分散空间支持。先扫描有效深度，再限制点预算，避免规则抽样漏掉大部分稀疏点。
2. 有界确定性 RANSAC 拟合面，拒绝混合面、非前向竖直面。输出相机到拟合面的**垂直距离**，不是任意像素的欧氏距离，更不是人体净空。原始深度是相机光轴 Z，通过有效内参及 pose 转换。
3. “可能较近”需要连续 3 次一致面；“相机接近候选”还需要跨帧位移、面距变化一致、旋转小、连续时间窗口。不会用神经深度波动或积分加速度推导距离。
4. 左中右候选各检查下部 6×6 网格的语义地面、有效深度、障碍覆盖、未知比例、各行连续性及水平面支持；多方向相近则弃权。始终保留人体朝向/净空未验证原因。

阈值在 `app/src/main/assets/ar_spatial_config.json`，版本 `sightguard.ar-spatial.v1`。ROI、6×6 候选划分、空间覆盖与拟合规则是当前研究实现，尚未真机标定。已有 PC Corridor 严格门禁和冻结 benchmark 未修改。

`ArSpeechPolicy` 独立事件去重、15 秒冷却、2.5 秒有限恢复，只有真实音频 `onStart` 才记作已提醒。研究提示低于正式风险/语义提醒，复用静音、音频焦点、停止和过期取消逻辑。已开始的声音不会只因播放中越过 500ms 起播期限而被切碎；场景失效/停止会取消。

## 时间与预览修复

- ARCore `Frame.timestamp` 的时间基准不能假设等于 Android capture timestamp。CPU 图像通过同一次 `Frame.acquireCameraImage()` 关联，拒绝重复图像，不人为加减固定偏移强凑时间戳相等。
- 只有 Camera2 声明 REALTIME 才把 `Frame.androidCameraTimestamp` 与 elapsedRealtime 比较；未知时间源拒绝几何结论。IMU 仅保守地否决快速运动，没有宣称完成未知时间源的精确融合。
- 源帧消费门限仍为 500ms；原始深度门限 100ms。跟踪丢失、场景变化、转向、代际变化取消历史/新语音，旧模型结果不能在失效之后复活。
- 初版直接拿 CPU 图像范围渲染 GPU 纹理，两者裁剪不同，造成画面两侧拉伸。现使用官方 NDC→texture 映射、按 GPU 比例留边；检测框另用 image→view 映射并裁剪到视口。实际红米截图已确认拉伸消失。未完成所有横竖屏组合的人工视觉验收。
- YUV 转换缓存尺寸/步幅并批量读平面；SegFormer 改为连续 buffer 读取、连续数组遍历，保留原 argmax/softmax和阈值。实际模型对照测试验证优化前后类别、坐标一致，置信度差异不超过 1e-6。

## 验证与性能

聚合记录：[`arcore_v05_validation.json`](arcore_v05_validation.json)。原始设备日志、截图只保存在本地 `android/deliveries/arcore-awareness/`，不提交私人场景到 Git。

- JVM：147 测试，0 失败/错误/跳过，其中 16 个 AR 几何/时效/语音策略/预览映射测试使用合成证据。
- 红米 instrumentation：最终真实模型与几何预览生命周期套件 4 测试通过；真实 YOLO/文件视频回归 4 测试通过；独立音频测试 1 测试通过（五条语音全部 onStart + completed，另检查静音及过期拒绝）。
- SegFormer 同输入 PyTorch 对照：最大绝对 logit 差 0.0001449585，argmax 一致率 1.0。仍使用原模型 SHA `85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f`。
- 五条离线 WAV 的请求→onStart：83 / 50 / 63 / 51 / 49ms，均在 500ms 期限内。**这些是资源播放测试，不是自动空间事件真机正例。**
- 非调试版 AR 采样：39 条主链日志，FPS 中位 11.61；处理耗时中位 78ms（不含采集到应用到达的年龄）。148 条语义/几何记录，源帧年龄中位 436ms、范围 343–492ms；语义耗时中位 156ms，后处理中位 10ms。PSS 抽样中位 459910KB，最高 507908KB；不是精确峰值内存。
- 后续 CameraX 采样 67 条：FPS 中位 11.31，处理耗时中位 86ms。独立 dumpsys PSS 599300KB。场景、运行时长与热历史不受控，**不能据此断言 ARCore 零开销或更快**，也不能把此前 23 FPS 当本轮对照基线。
- AR 到帧本身约 240ms 的年龄不能从处理延迟中省略。早期调试版出现大量 >500ms 的结果并弃权；优化后的非调试版窗口才达到上述时效。没有延长 TTL。
- ADB UI：静音/恢复、主动停止后前后台不自行重启、重新开始、设置关闭/开启 AR、资源释放已检查。最终留下正在感知、未静音、实验开关开启的版本。未重做完整 TalkBack/大字体验收。

## 失败记录和证据边界

1. 用户说明起初手机没有拿起：低光/无有效特征那段只能作为运行与弃权测试，不能作为墙面测距失败的受控样本。
2. 后续见到 TRACKING、零散深度，以及纯旋转/过快移动等拒绝原因；没有标定墙距和动作标签，不计距离误差或接近准确率。
3. 最终非调试版记录虽新鲜，但高置信度墙面深度不足，正确 ABSTAIN。静止、低纹理、遮挡及旧 raw-depth 无法支撑新测距；不能用降置信度/延长深度寿命来演示成功。
4. 初版严格比较 CPU/AR/Android 三种 timestamp 导致主链丢帧，已按 API 关联契约修复；未将未知偏移伪装成同步。
5. 初始 instrumentation 从后台启动被 HyperOS 阻止，保留失败日志；改用已授权 instrumentation shell `am start`，未关闭系统权限机制。
6. 组合音频回归发生音频焦点中断，未满足 completed，保留失败日志。独立执行最终五条全部通过；产品继续遵守焦点丢失取消。
7. 尚未验证：数值距离误差、受控相机接近与横移/旋转混淆率、真实候选区域、连续长时间发热/耗电、所有机型兼容性。无安全路径能力。

下一步优先采集少量带卷尺真值和明确动作标签的安全场景，检查 raw depth 覆盖、内参与语义对齐，再决定是否开放数值距离及更广泛的自动接近播报。若几何支持长期不足，继续使用已验证的墙面存在提醒。

## 构建、安装、回退

本轮起点 SHA `fa2315595da6f2396487b5955a6e598a03c1c133`。旧版 APK 保留，不覆盖。新工作在独立分支；最终 PR 仅提交本轮差异，不合并 master。

```powershell
. ./android/tools/task_environment.ps1
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:assembleRelease
# 当前 SDK 的 adb；先用 adb devices 选择已授权设备
& D:/Android/Sdk/platform-tools/adb.exe install -r android/deliveries/arcore-awareness/SightGuard-0.5-arcore-experimental.apk
```

非调试 release 为**本地实验分发**，使用原开发签名以支持覆盖安装，不是生产商店签名；versionCode 6。实测安装后执行过标准 `adb shell cmd package compile -m speed -f org.sightguard.app`，性能复现需记录该步骤。

- 新 APK：`android/deliveries/arcore-awareness/SightGuard-0.5-arcore-experimental.apk`，SHA256 `adc3912d383dfe5e8bcf7e3414b438faea10e1ca8cd89874fa9b9db709775384`。
- 同源调试 APK 及 SHA 见聚合记录；不要用调试版帧率代表非调试版。
- 首选回退：设置关闭“实验性空间提醒”，不卸载、不清除用户数据，即恢复 CameraX、原模型及语音。
- 保留稳定 APK：`android/deliveries/metric-corridor-research/baseline/SightGuard-0.4-stable.apk`，SHA256 `6671a81d740d5a12087f7af6eb8796b7565f8c28add75f59ffbcc5f022753fd7`。旧实验版 v5 SHA `72ad507807ffb54b570a11996da38cc4c588ce5b78f83376575fcf316ce88d0a`。
- 若设备允许开发包降级，可用 `adb install -r -d <旧APK>`；若系统拒绝，不自动卸载/清数据，先用开关回退。本轮没有实际执行破坏性降级。

## 官方依赖与隐私

- [ARCore 支持设备列表](https://developers.google.com/ar/devices/)：Redmi Turbo 4 Pro 支持 Depth API。
- [ARCore Android SDK 1.56.0 官方发布](https://github.com/google-ar/arcore-android-sdk/releases/tag/1.56.0)，SDK 代码 Apache-2.0。Google Play Services for AR 是独立服务，遵守其条款/隐私政策；应用设置提供政策入口。
- 官方运行时 APK SHA256 `8c6dac408ff9320515ce105fb17e3f76087762b6b7ad05b6da49030fee3839b6`；安装前签名验证 Google 证书 SHA256 `6e913f65ea53321ead6932ff777e1b7f94bbba7e2b0dafdd8494b100101e3797`。
- [Frame 时间与图像接口](https://developers.google.com/ar/reference/java/com/google/ar/core/Frame)、[Raw Depth 说明](https://developers.google.com/ar/develop/java/depth/raw-depth)、[官方预览映射示例](https://github.com/google-ar/arcore-android-sdk/blob/main/samples/hello_ar_java/app/src/main/java/com/google/ar/core/examples/java/common/samplerender/arcore/BackgroundRenderer.java)。
- SightGuard 合并后的 manifest 未新增 INTERNET 权限；本轮未上传用户图片/视频/原始设备日志。SDK/服务下载需要网络，不等于应用视觉推理需要云端。未新增研究数据集，复用仓库原有已注明来源的模型/测试素材。
