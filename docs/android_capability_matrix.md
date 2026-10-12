# Android 0.4 / PC 能力差异

核对日期：2026-10-12，Android 稳定基线 4a1fdfb、米制研究 b3d0aeb。参数与测试范围必须按版本报告，不将单张同输入对照推断为整体精度等价。

## 检测、跟踪与风险

- PC：YOLO26n / BoT-SORT，默认配置 640、阈值 0.25、限定 COCO 类别；可改配置进行评测。
- Android：YOLO26n ONNX 320、阈值 0.35、COCO80；轻量几何跟踪和短时恢复，不是完整 BoT-SORT。
- 规则评分与事件生命周期有历史同输入对照；移动端另加风险相关性、播报冷却、方向迟滞、资源 / 会话策略，不能声称所有运行行为相同。
- PC 已结束事件仍留在历史字典；Android 已移除 RESOLVED 事件。长会话问题见[清单](product/known-issues.md#p0-04-pc-长会话事件保留未有界化)。

## 环境语义与空间证据

- Android SegFormer MiT-B0 / ADE20K 256：正式环境事件自动运行，默认墙面 / 楼梯 / 杆柱 / 栏杆；图层显示不是推理开关。
- STEP 原始 121 类和事件实现保留，真实台阶 / 路缘识别不足，自动播报关闭。
- PC Path Relevance / Grid / Corridor：独立研究；PC 默认实时规则链没有自动接管 Android 环境语义策略。
- 两端已有相对 Depth Anything V2 Small；相对深度不是米制测距。Android 原 Depth 真机约 0.9 秒，过期结果不能用于新语音。
- Android 已有 15×15 Grid 和 IMU 运动质量研究；没有安全通道结论，UNKNOWN 时钟不作精确融合。
- 新米制墙距、视觉接近与局部候选首先在 PC 研究。Android 只有默认关闭的米制适配；局部 Corridor 评估器仍仅 PC。

## 声音、UI 与文件

- Android：中文左 / 前 / 右固定离线语音与动态离线中文 TTS 支持。真实起播由 onStart 确认，缺少动态 voice 时描述只显示文本。
- PC：可选 OpenAL HRTF，需显式准备 / 启用；网页试听和演示方向与正式事件分离。
- Android：原生简洁无障碍主界面，前台自动感知、主动停止和静音；研究工具在设置中。
- PC：FastAPI 产品演示页和按需 Gradio 研究台，是开发 / 评测工作台。
- Android 文件分析是抽样解码，时间是请求位置估计；PC analyze 是完整帧记录，精确时间以记录的来源类型为准。二者不能直接混算延迟和召回。

## 验收与仍缺少的能力

0.4 稳定包有 nova12 真机模型 / 音频 / 短生命周期回归；TalkBack、200% 字体专项在 AVD。实验包后续 ADB 安装成功，但新米制 Android 推理、FPS、内存和 CameraX 时效未验证。

两端都没有经过部署域验证的米制墙距播报、身体接近、净空宽度、安全路线或自动绕障。具体证据和失败见[当前状态](product/status.md)、[0.4 验收](../android/docs/accessible_awareness_v04.md)和[研究记录](../android/docs/metric_corridor_research.md)。
