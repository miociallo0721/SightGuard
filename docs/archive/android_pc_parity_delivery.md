> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard Android 0.2 — PC 核心体验对齐交付

日期：2026-10-11。工程：`D:\program\SightGuard\android`。本轮由主代理负责风险事件核心、共享 VisionStage、Compose 集成、构建与 AVD 验收；GPT-6 Luna 子代理完成独立的特征评分、语音与视频任务。

## 可安装成果与三个独立状态

- 最新 debug APK：`D:\program\SightGuard\android\deliveries\parity-final\SightGuard-0.2-parity.apk`，123,663,493 bytes，约 118 MiB。Android API 26+；arm64-v8a、armeabi-v7a、x86_64。
- SHA256：`c95b69beb9fe0c7720e189cb34bc8aec4254c059a3b26d6eefe5873289c99d25`。
- **构建交付：通过。** 实际执行 assembleDebug；最终日志 `android/parity-validation/delivery-build.log` 为 `BUILD SUCCESSFUL in 55s`，并用 apksigner 验证签名、AVD 实际安装。APK badging 无 INTERNET 权限。
- **Android AI 视觉感知：通过 AVD。** YOLO26n 真实照片、CameraX 连续帧和真实街景 MP4 实际推理；可选 SegFormer 也实际执行。没有模拟检测输出。
- **声音提醒：固定本地中文音频实际播放通过。** Android16 AVD 缺少离线中文 voice，平台动态中文 TTS 尚未通过。应用检查 voice 的中文语言、网络要求和未安装标记，不联网；缺少时清楚显示固定音频降级。79 条短句来自构建时 Microsoft Huihui Desktop，清单与 SHA 在 `android/validation/local_audio_manifest.json`。

最初已验证 MVP 仍保留在 `android/deliveries/mvp-verified/SightGuard-0.1-mvp-verified.apk`，SHA `ff9c225798856555173a4b3de4916e71868636526b0ecd6e4f01197c7f9e6469`。P0/P1 和 P2 阶段 APK 分别保留在 `deliveries/parity-p0-p1/`、`deliveries/parity-p2/`，未覆盖原副本。

## 相比 MVP 实际新增

纯 Kotlin 移植 PC 的 FeatureExtractor、RiskScorer、AssessmentEngine、事件生命周期与 AlertScheduler：包含质量、连续帧确认、迟滞、丢失恢复、ID 变化、升级和感知受限。CameraX 与文件分析共用 VisionStage；风险核心不依赖 Android UI 或模型运行时。

移动端另外采用 2.5 秒事件恢复窗口、每帧最多 8 项调度候选、3 秒普通提醒冷却、方向迟滞、同类同方向谨慎合并和近期强几何关联。提醒在实际 `onStart` 后才提交“已提醒”，被拒绝、静音、过期的候选可重试；重要新风险与升级能越过普通冷却。记录包含原始事件、请求、接受、开始播放、抑制与取消。

轻量跟踪增加时间间隔相关速度预测、一对一互相最佳匹配、短时恢复和歧义弃配；**不等价于 BoT-SORT**。同帧强重叠重复框只在提醒关联中合并，界面仍保留原始模型输出。

新增 SAF 本地 MP4 分析、播放/暂停/时间轴；跳转清空框、声音和空间证据并重建事件；到末尾后再次播放从零开始。画面、检测框与语义层绑定参与分析的同一实际解码图像。MediaMetadataRetriever 不返回逐帧 PTS，时间明确标为“请求位置估计”，不是 PC 逐帧回放的严格时间等价；抽样步长至少 100 ms，不承诺完整 30 FPS 分析，也不播放源视频音轨。

原生深色竖屏界面增加红框与置信度、风险、感知状态、输入模式、实际完成帧 FPS、延迟和图层开关。仅前台运行，切后台停止并释放，回来需主动重新开始。

完整能力矩阵：[源码内能力对照](android_capability_matrix.md)，本机另有 [交互Canvas](D:/Android/codex-canvases/d-program/SightGuard-android-parity.canvas.tsx)。未移植 DemoGuidance、HRTF、Depth Anything V2、Spatial Grid 和 Corridor；没有安全路径或人工绿色 Mask。

## 模型、运行时与真实对照

统一运行时 **ONNX Runtime Android 1.24.3 CPU**，没有同时集成 LiteRT。

- YOLO26n 原权重 SHA：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。APK ONNX SHA：`e093d7e1805d63cc1b671c333ce0b48b5cd33f70f64ea16c9b5bd713e33242cf`。实际输入 float32 RGB/255 NCHW `[1,3,320,320]`，输出 `[1,300,6]` 的 xyxy/score/class，已带 end-to-end TopK，不另加其他 YOLO 格式的 NMS。中心 letterbox，pad114，confidence0.35，COCO80。
- 同一真实 bus.jpg，Android 与 Python 同 320 输入匹配 5 个目标，IoU 0.99397–0.99898，最大 confidence 差 0.01150。报告 `android/validation/android_model_validation.json`。**PC 默认 640、0.25、类别筛选，与本 APK 的默认检测行为不同；该单图对照不是精度 benchmark。**
- SegFormer MiT-B0 ADE20K 原 checkpoint SHA：`8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`。APK ONNX SHA：`85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f`。真实归一化输入 `[1,3,256,256]`，输出 logits `[1,150,64,64]`。Android 相同张量对 Python 类别一致率100%，最大绝对误差0.0001182556；原图推理确实产生地面类别。最终设备测试加载463ms、同张量推理389ms、原图处理419ms，进程PSS从168294增至219970KiB（包含缓冲等，非独立模型内存）。实际暖机并行推理常见180–408ms，冷启动更慢；完整记录见 `android/segformer-validation/android_segformer_validation.json`、`delivery-camera-live.log`，桌面报告另存同目录。
- SegFormer 默认关闭，可独立启停，每2秒最多一帧，独立单线程，不积压，不阻塞基础串行 YOLO；实时证据500ms过期，不在新视频时间点显示旧结果。只显示 floor/road/sidewalk、confidence≥0.35 的真实琥珀色地面语义。没有地面为 ABSTAIN，失败为 UNKNOWN，不进入风险评分、不代表安全区域。Android 256/bilinear 与 PC 默认512/INTER_AREA预处理不同；相同张量数值通过不等于整套 PC 空间功能等价。

SegFormer 来自仓库已有 OpenMMLab 权重；单独 checkpoint 再分发许可尚未独立核实，沿用 `docs/research/path_relevance_v1.md` 的准确记录。模型、视频和测试 tensor 不提交 Git。导出依赖是独立环境（Ultralytics8.4.174、Torch2.14.1 CPU、ONNX1.19.1、ORT桌面1.24.4、Transformers5.19.0），没有改变 PC uv.lock。

## 测试证据

- Kotlin JVM：**48 项通过，0 失败**。覆盖方向/迟滞、持续静默、消失再出现、短遮挡、仲裁、重要新风险绕过冷却、过期、重启、TTS失败/静音/停止、轻量跟踪和坐标映射。最终lint为0错误、18项warning（更新建议、竖屏和API写法等），报告在Android build/reports中。
- Android instrumentation：9项，涵盖两项共享 PC 对照、真实 YOLO、真实 SegFormer、RGBA crop/stride/四种旋转、三项 MP4 解码与真实 VisionStage。设备日志：`android/parity-validation/delivery-device-instrumentation.log`。
- PC/Kotlin 对照：200帧先前真实 CameraX 检测记录 + 7组明确标记的合成状态场景；逐项比较特征、质量、分数、等级、事件状态/时间/退出/恢复、AlertEvent和finish，数值容差1e-9，JVM与Android均通过。fixture `android/app/src/sharedTest/assets/parity_reference.json` 含配置、原trace哈希和模型哈希。**证明这些输入下默认核心语义一致，不证明不同检测/跟踪/移动端配置下与 PC 整体等价。**
- 约90秒固定真实照片 CameraX 连续检测，最后日志到第1321帧；4个目标形成3条首次实际播放（公交车、合并左侧两名行人、右侧行人），后续稳定持续无重复。暖机后每10帧记录样本的FPS中位数15.59、耗时中位数61.5ms；是此AVD片段，不是手机性能。
- 最终SHA对应APK另跑约155秒、日志至1711帧：3条首次提醒和1条左侧风险升级，之后持续静默，未超期播放或崩溃。后约58秒启用空间模块，YOLO持续工作，关闭空间模块单独释放成功。顺序片段的完成帧FPS样本中位数：默认12.47、空间启用11.71；耗时76.5/81.5ms。进程PSS167493/263396KiB。属于单次AVD观察，受冷启动和主机负载影响，不是严格消融或手机速度下降比例。记录在 `delivery-live-audit.json` 和 `delivery-camera-stop.log`。
- 静音状态重新开始：有真实检测和事件，欢迎音被 `detail=muted` 抑制，没有播放；取消静音后未播首次提醒正常发声。真实虚拟房间画面检测为零约46秒，无风险播报。停止和后台可见输入/模型/spatial释放及排队音频取消。日志 `final-camera-live.log`、`final-camera-stop.log`、`final-background-camera.txt`、`final-background-audio.txt`。
- 真实街景 MP4 由 SAF 选择、480×360实际解码、真实检测/声音/暂停/跳转/末尾重播均已验证；跳转2585/7394/19999/0ms清空Overlay，新的frameIndex0不冒充已确认风险。暂停7394ms同帧SegFormer产生1107个支持的地面单元。日志 `final-video-live.log`。
- `final-playback-audit.json` 保留实际播报、跳转、取消与采样性能；已记录样本没有超过deadline才开始的音频，也未发现FATAL崩溃。有效测试未删除或弱化；辅助Python脚本Ruff通过。Python PC回归仍为MVP阶段567 passed、27 skipped，本轮未改动其核心。

验收设备仅 **medium_phone AVD，Android16/API36 Google Play x86_64**。2 cores、540×1200、host GPU、无窗口/快照；请求1.5GB但镜像实际提高至2.5GB RAM。飞行模式下验证，应用无 INTERNET 权限。用户报告旧MVP在**红米Turbo4Pro、Android16、语音开启约23FPS**；该记录不是本代理真机测量，新版真机FPS/发热/长期稳定性/离线中文voice仍待测。空间模块会争用CPU，当前AVD记录不足以给出手机性能影响比例。

相机截图：delivery-camera.png：`../../android/parity-validation/delivery-camera.png`（历史本机产物，不随仓库分发）。文件截图：delivery-video.png：`../../android/parity-validation/delivery-video.png`（历史本机产物，不随仓库分发）。真实文件及语义截图：final-video-spatial.png：`../../android/parity-validation/final-video-spatial.png`（历史本机产物，不随仓库分发）。零目标截图：final-empty.png：`../../android/parity-validation/final-empty.png`（历史本机产物，不随仓库分发）。街景来源为 Uwebart 的 [Traffic control in Mexico City](https://commons.wikimedia.org/wiki/File:Traffic_control_in_Mexico_City.ogv)，CC BY-SA3.0，裁剪与转码记录在视频旁 metadata.json；仅单个20秒片段，不是安全验证数据集。

## 限制与复现

尚未设备级注入验证：无摄像头、摄像头占用、模型损坏、实际平台中文TTS失败和音频焦点竞态。拥挤快速街景中几何跟踪仍会生成新事件，AVD画面抖动也可触发启发式风险升级，不能将本次固定场景去重结果推广到所有街景。约数分钟AVD检查不代表手机长时间测试。仅辅助感知研究原型，不宣称碰撞预测、米制距离、导航或安全保证。

构建/安装/模型复现命令见 `android/README.md`。SDK/Gradle/uv/导出/AVD/任务临时产物全部物理位于D盘；Canvas原C盘目录为D盘junction。验收后关闭AVD降低内存。19:10检查C盘约4.6GiB可用，系统 `C:\pagefile.sys` 文件约21GiB、18:44更新；缓存迁移入口仍指向D，未调整系统分页或重启。Blender占用的旧uv部分按用户要求保留。未改AGENTS.md、冻结Benchmark或用户已有PC修改。提交和PR按随后用户指令单独执行，不自动合并。

本文的APK、截图、日志、模型及实验报告路径为本机交付证据，不提交Git；源码包含复现脚本、版本与哈希、必须的运行资源以及回放测试fixture。GitHub审阅请使用源码内能力对照，不能将本机绝对路径当作在线下载链接。
