# 架构

SightGuard 将模型推理、证据管理、提醒策略和界面分开。Android 与桌面共享问题域和回放契约，运行管线与跟踪实现各自独立。

## Android

`CameraX / 本地视频 → VisionStage → 目标跟踪与风险事件 → SpeechController`

低频辅助链：`同源图像 → SegFormer / 可选 Depth → 环境证据与 Spatial Grid → 环境事件 / 局部空间候选`。

- 目标主链优先；辅助任务不积压，不通过延长 TTL 掩盖过慢推理。
- `SpatialFrameKey` 绑定会话、图像尺寸、源帧与时间。过期或来源不匹配的证据不触发新提醒。
- 环境事件独立于既有风险评分，经统一声音队列调度；实际播放回调与事件提议分开。
- 候选带只消费已确认的图像空间证据，使用与预览一致的映射；不生成世界坐标路线。
- 普通界面负责状态与核心操作，研究参数留在开发者工具。

代码入口：[MainActivity](../android/app/src/main/java/org/sightguard/app/MainActivity.kt)、[候选评估](../android/app/src/main/java/org/sightguard/app/LocalSpaceCandidate.kt)、[声音调度](../android/app/src/main/java/org/sightguard/app/SpeechController.kt)。

## 桌面

`CLI / UI → pipeline → perception → FrameRecord → risk / decision → AlertEvent`

`perception` 适配 YOLO26n / BoT-SORT；`risk` 与 `decision` 不依赖模型或 UI 对象。`evaluation` 管理独立标注与对照，`audio` 管理可选空间音频。已保存的感知结果可以重放，不必重复模型推理。

历史接口与数据格式见 [PC 数据契约](reference/pc-data-contracts.md)。研究模块的深度、运动和空间证据不会自动成为正式提醒，详见[研究索引](research/README.md)。

## 资源与数据

模型、原始视频、构建缓存和本机交付物不提交 Git。模型清单、配置、锁文件、必要的小型固定回放样本和可复现工具保留。JVM 和 Android 对照测试共享 `android/app/src/sharedTest/assets/`，避免两份大数据漂移。
