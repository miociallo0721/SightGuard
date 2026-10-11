# Stage 4A：Motion Quality 与 Temporal Evidence v1

本阶段以真实 RGB 相邻帧建立 CPU 图像运动研究接口，复用已保存的 YOLO/BoT-SORT 检测、Spatial Grid v03 和 Corridor v01。二维图像运动不是身体速度或绝对相机姿态；所有 `absolute_camera_pose_status` 均为 `UNKNOWN`。模块没有生产提醒、音频、导航动作或 live camera 接入。

## 算法选择与参考

主算法为 Shi–Tomasi + 双向 Lucas–Kanade + RANSAC partial affine。LK 适合连续短间隔、局部纹理可跟踪的帧；双向误差减少错误匹配。ORB 更适合较大视角变化或重定位，但描述符匹配成本和重复纹理歧义更高。本阶段优先相邻帧，不加第二条主算法。Partial affine 以二维旋转、统一尺度和平移描述全局图像变化，参数少于全仿射/单应性；强视差、遮挡、明显透视变化时允许失败，不以额外自由度制造可信估计。

参考 OpenCV 官方 [LK 文档](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)与 [RANSAC/estimateAffinePartial2D 文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

Roana 固定版本 `7da28b7dd127f002986711ea62249b73de723739` 已从本地审计副本核验：

- [MotionQuality.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/MotionQuality.kt)：参考质量与原因分开保存的接口。其缺样本时 STABLE/motion_unavailable 的规则没有沿用；IMU pitch/角速度阈值也不移植到图像运动。
- [CorridorStateMachine.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorStateMachine.kt)：参考失效需要打断历史判断的边界；未实现其 STOP/方向状态机或帧计数确认。
- [CorridorPipeline.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorPipeline.kt)：该固定版本管线未消费实测 MotionQuality，也未给出量化 TTL。不能把其整体架构当成已有运动融合验证。
- [MainActivity.kt](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/MainActivity.kt)：参考原生帧时间和 frame-loss 的区分。本实现使用视频 PTS，未复制上游实现。

没有复制、翻译 Roana AGPL 源码、测试或 fixture；新增算法与合成测试独立编写。

## 实际估计过程

1. 校验 video/stream、相邻索引、图像格式/尺寸、实际 dt、视频 PTS 与检测结果身份。runner 连续解码，解码 PTS 与已保存 tracks 时间必须在 0.01 ms 内一致；不采用固定 FPS 修复异常时间。
2. 灰度图宽度最多 640 px，不放大小图。检测置信度至少 0.5 的框，包括静态可能性很高的椅子，按原图扩展 12 px；前后两帧都排除框。未检测区域始终只是候选背景。
3. 最多 500 个 Shi–Tomasi 点；前向/后向 LK、状态、图像边界、当前排除区域、LK error ≤30、前后向误差 ≤1.5 work px 共同过滤。
4. RANSAC partial affine：阈值 3 work px、2,000 iterations、confidence 0.99、固定 RNG seed 41。CLI 在自己的进程设置 OpenCV 1 线程；不修改运行中的用户服务。
5. 保存匹配数、内点数/比例、内点残差 median/P95、全匹配残差 P95、4×3 内点网格覆盖、可用区域比例及 Laplacian variance 代理。
6. 把变换恢复为原图像素坐标。运动幅度取变换预测的四角与中心共五点的位移范数中位数，除以原图对角线和实际 dt；它不是所有点流速的中位数，也不是米/秒。旋转是二维图像角度，尺度是图像尺度比。

失败允许保留已算出的诊断数值，但 `estimation_status=UNKNOWN` 的数值不能用于有效运动窗口。最多保存 80 对匹配点供示意；无箭头或导航含义。

## 质量规则

全部参数见 `configs/motion_quality.yaml`，版本 `sightguard.motion-config.v1`。下列阈值是未校准工程选择，不代表现实角速度或安全概率。

- 无前帧、解码失败、时间/序列不一致、dt 不在 10–120 ms、严重成像退化代理、可用背景低于 20%、特征少于 40、有效匹配少于 30、RANSAC 失败、内点少于 30/比例低于 0.35、内点覆盖低于 0.25、内点残差 P95 大于 4 work px：UNKNOWN。
- 旋转绝对值大于 20°/frame、尺度不在 0.85–1.15 或预测位移大于 0.25 对角线/frame：UNKNOWN，而不是把不可信变换称为快速相机运动。
- 有效估计但匹配少于 80、内点比例低于 0.6、覆盖低于 0.5、残差 P95 大于 2 work px、Laplacian variance 低于 40，或连续有效估计的速率变化大于 0.75 diagonal/s：LIMITED。
- 其余有效估计按 diagonal/s 分类：≤0.08 STABLE_IMAGE_MOTION，≥0.5 RAPID_IMAGE_MOTION，中间 MODERATE_IMAGE_MOTION。

`VALID` 可以伴随 LIMITED；有效二维拟合不等同足够可靠的复用证据。高内点比例也可能来自共同运动的前景、重复纹理或局部区域。单个二维模型无法唯一分解相机自运动与独立物体运动；检测漏检、静态框误排除、滚动快门、视差、俯视方向变化与遮挡均可能破坏解释。

## 因果窗口与 Freshness

对每个原 2 Hz frame key，聚合 `(t−500 ms, t]`。只使用截止 t 的同 video/stream 估计，并验证前帧身份、连续索引与 dt 一致性。帧间区间需完整落在窗口内；首帧、不完整历史、未来估计不补零。保存 source indices、文件引用、覆盖时间、最大缺口、有效比例和 P50/P95。

有效估计比例至少 0.6、有效时间覆盖至少 0.8、最大缺口至多 120 ms，并有运动数值，否则 UNKNOWN。达到这些门槛但含 UNKNOWN/LIMITED 或部分覆盖时为 LIMITED；干净窗口才按 P95 分类。这是保守规则，真实实验的所有窗口均 LIMITED/UNKNOWN，尚不适合生产复用。

Freshness 在本轮使用模拟消费时间 `source.timestamp_ms + delay`，delay 为 0/100/250/500/1000 ms：

- 来源 key/stream 不一致、消费时间非法/早于来源或原 Corridor ABSTAIN：INVALID。
- 窗口 UNKNOWN/LIMITED：UNKNOWN，禁止研究复用；不自动继承旧候选。
- 其他窗口 TTL：stable 500 ms、moderate 250 ms、rapid 100 ms；age ≤ TTL 为 FRESH，超过为 STALE。

此 evaluator 只使用来源时刻已知的运动历史，没有偷看消费时刻以后的图像。它不能证明 TTL 内环境未发生变化；未来实时消费者仍需当前质量、新障碍和失效通知。模拟消费年龄不是推理、音频或端到端延迟。

## 契约、模块与使用

- `scene/motion/schema.py`：不可变配置/帧级/窗口契约。FrameKey 复用现有接口，不引入世界坐标。
- `estimator.py`：独立 CPU 估计；输入 MotionFrame 包含图像、已保存 FrameRecord、流身份、错误与 provenance。引擎不保存跨视频状态，previous_estimate 只能参与同一相邻流的突变检查。
- `temporal.py`：窗口、Freshness 与候选变化诊断；不依赖 UI、risk、decision、Torch、YOLO 或音频。
- `pipeline/motion_quality.py`：父结果哈希预检、连续解码、双算法对照、同 key 关联、版本化 JSONL、运行错误/性能记录。错误运行保存 failed metadata，输出目录禁止覆盖。
- `visualization/motion_quality.py/.html`：独立真实 RGB 只读 viewer。未遮罩模式仅切换帧级对照；窗口和 Freshness 始终来自主遮罩估计，来源引用可展开检查。

序列化版本：`sightguard.frame-motion.v1`、`sightguard.motion-window.v1`、`sightguard.freshness.v1`、`sightguard.temporal-corridor.v1`、`sightguard.motion-batch.v1`。FrameMotionEstimate 包含前后帧身份、dt、拟合状态/质量、数值/nullable 数值、算法配置和 provenance。窗口包含时间范围、有效性、分位数和源引用。Temporal 行保留完整 `source_corridor`，新增 `change`、`motion_window`、模拟 `freshness`，不写回原文件。

Temporal 分类分开记录最低成本候选交换、BEST→TIED、TIED→BEST、ABSTAIN/恢复、近成本排序交换、区域证据变化和资格门禁变化。best 身份含 null 的变化不能直接称为方向抖动。分项归因是同名图像候选的成本增减描述，绝非世界空间对应或因果解释。

Temporal Shadow 默认关闭且不实现，开启配置会明确报错；缺少经过验证的跨帧候选对应和独立时序通行真值，不使用历史候选压制新障碍。旧严格 D 组保持姿态 UNKNOWN 全弃权。本轮结束后不继续开发导航状态机。

复现与真实结果见 [阶段验证报告](temporal_corridor_validation.md)。模型、真实视频和生成输出不提交 Git，依赖锁未修改。
