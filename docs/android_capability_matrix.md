# Android 0.2 / PC 能力对照

来源：2026-10-11仓库实现、48项JVM测试、9项Android16/API36 AVD设备测试及本机验收。下述对照限定所测输入；没有最终版本的真机测量。

- **Detection / FrameRecord：已实现，时源有差异。** Kotlin保留像素框、类别、分数、时间与状态契约；视频时间是请求位置估计，不是解码PTS。
- **YOLO26n：真实推理已验证，默认行为不同。** Android ONNX320/0.35/COCO80；PC默认640/0.25/类别过滤。同320真实照片的目标级对照通过，非整体精度等价。
- **BoT-SORT：需要移植。** Android为互相最佳几何关联、速度预测、短时恢复、歧义弃配的轻量跟踪器。
- **Risk Features / Scoring：同输入对照通过。** 纯Kotlin特征、质量、加权评分、迟滞；200帧实际CameraX保存记录和7组合成状态场景的全字段容差1e-9。
- **AssessmentEngine / EventManager：同输入对照通过。** 候选、激活、丢失、恢复、退出、ID变化、会话隔离及事件时间；移动应用另设2.5秒恢复窗口。
- **AlertScheduler：默认核心对照通过，移动交付有差异。** Android额外冷却、合并、方向迟滞、优先级、期限；实际音频onStart之后才提交已提醒状态。
- **DemoGuidance：需要移植。** 当前只展示风险和感知状态，不给绕行或安全导航建议。
- **Spatial Audio：行为不一致。** Android左/前/右中文；尚无PC空间音频研究后端和HRTF。
- **SegFormer MiT-B0：真实Android模型验证通过，预处理不同。** 256输入、默认关闭、独立低频线程和过期机制；相同张量argmax100%、最大误差0.000118。PC默认512/INTER_AREA；图层仅表示地面语义。
- **Depth Anything V2 Small：需要模型转换。** 本轮仅选择SegFormer验证，没有同时集成深度模型。
- **Spatial Grid / Corridor：需要移植。** 当前没有这些研究算法，没有人工安全Mask或路线。
- **CameraX / ONNX Runtime：已实现且AVD验证。** KEEP_ONLY_LATEST、后台串行推理、前台会话、旋转/坐标映射、停止与后台释放。ORT Android1.24.3 CPU。
- **本地文件分析：已实现且AVD验证，回放方式不同。** SAF、兼容MP4、同帧检测Overlay、暂停/跳转/末尾重播；抽样分析而非PC完整逐帧研究回放。
- **中文TTS / 本地音频：降级播放已验证。** 平台离线voice检查已接入；AVD缺中文离线voice，实际播放79条固定预合成片段，动态中文TTS未验证。
- **真机性能：当前未知。** 用户报告旧MVP红米Turbo4Pro/Android16/语音开启约23FPS；新版仅AVD，不能据此推算新版手机FPS或空间模型开销。

模型SHA、实际测试范围和限制见 [交付记录](android_pc_parity_delivery.md)。本机交互Canvas保留于D盘，不随源码发布。
