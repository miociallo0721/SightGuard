> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard Path Relevance v1 任务报告

日期：2026-10-11

状态：已完成离线研究型 Shadow 原型与八段视频验证；尚未启用生产提醒策略。

## 任务目标

为现有风险事件评估补充语义地面区域关系，研究检测目标与画面下方地面区域的图像空间交叠程度，并观察它是否能为后续人工复核提供信息。本阶段限定为 PC 离线回放，复用已有检测、跟踪和风险结果；不实现导航、路径规划、主动绕障，也不替代盲杖、导盲犬或定向行走训练。

## 实施内容

- 加入 SegFormer MiT-B0 / ADE20K 场景分割适配器，读取本地 checkpoint，提取 `floor`、`road`、`sidewalk` 类别，筛选从图像下方延伸的候选语义地面连通区域。
- 以检测框内地面像素比例和框下部地面支持比例计算对象级 Path Relevance；低质量场景、无有效区域或无效检测框返回 `UNKNOWN`，不把未知当作低风险。
- 将结果保存为可序列化的场景帧、对象和 Shadow JSONL，并提供离线回放及只读 Research Console 展示。
- 建立 A/B/C/D 离线比较：原 Baseline、静态几何 ROI、SegFormer Path Relevance、仅记录候选建议的 Shadow。Baseline alerts 保持只读；生产决策变更为零。
- 研究了 Roana 的显式候选通行区域建模思路；没有复制或移植其代码。实际推理使用 OpenMMLab 发布的 SegFormer checkpoint，并单独记录 checkpoint 许可证尚未核实，不以仓库许可证推定权重可再分发。

实现边界、公式、数据接口和许可证来源见 [Path Relevance v1 设计说明](D:/program/SightGuard/docs/path_relevance_v1.md)。

## 验证数据与方法

权威批次为 `outputs/path_relevance/v04/`。输入是同一原始拍摄中的 8 段 UrbanEgo development 视频：总时长 120 秒、20 FPS、720p、2,400 原始帧。每段按 0.5 秒间隔取样，共 240 个场景帧。复用已保存的 YOLO26n / BoT-SORT tracks 和 Risk Engine 结果，没有重新运行检测器，也没有修改 Risk 阈值。

运行使用 Python 3.12.15、PyTorch 2.10.0+cu130、CUDA 13.0、Transformers 5.19.0、NumPy 2.5.3、OpenCV 4.14.0.94，以及 NVIDIA RTX 2050 Laptop GPU（4GB）。模型 checkpoint SHA-256 为 `8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`，大小 15,072,692 bytes。批次元数据记录了模型、输入、配置、软件和设备信息。

## 结果摘要

8/8 段处理完成。240 个采样场景帧中，238 帧输出有效候选地面区域，2 帧为 UNKNOWN。共生成 1,123 个对象观察，其中 741 个有定义的 Path Relevance 分数，加权均值为 0.319；等级分布如下：

- HIGH：125
- MEDIUM：376
- LOW：240
- UNKNOWN：382

原 Baseline 共 257 条 AlertEvent（HIGH 120、MEDIUM 137）。八段视频的 `alerts.jsonl` 哈希在实验前后全部一致。Shadow 建议按目标观察计数：保留 Baseline HIGH 468、Path UNKNOWN 时保留 Baseline 219、保留高相关候选 67、人工复核 366、仅研究性降低优先级 3。上述数字不是独立提醒数量，也不是已过滤的 AlertEvent；三条研究性降优先级建议不能解释为减少了三次误报。

| 视频 | Baseline 事件 | 有效场景帧 / 30 | 对象观察 | 已定义 PR 均值 | HIGH / MEDIUM / LOW / UNKNOWN | 样本处理均值（ms） |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| V01 | 33 | 30 | 143 | 0.296 | 13 / 65 / 31 / 34 | 67.3 |
| V02 | 55 | 30 | 211 | 0.378 | 31 / 95 / 38 / 47 | 50.9 |
| V03 | 34 | 30 | 144 | 0.073 | 0 / 15 / 75 / 54 | 51.4 |
| V04 | 20 | 30 | 108 | 0.070 | 0 / 1 / 7 / 100 | 42.6 |
| V05 | 36 | 30 | 140 | 0.288 | 17 / 34 / 43 / 46 | 55.9 |
| V06 | 25 | 29 | 122 | 0.366 | 12 / 64 / 6 / 40 | 51.2 |
| V07 | 48 | 30 | 188 | 0.477 | 44 / 73 / 10 / 61 | 52.1 |
| V08 | 6 | 29 | 67 | 0.255 | 8 / 29 / 30 / 0 | 61.0 |

V04 表中的 PR 均值仅对有定义分数的对象计算，UNKNOWN 未按 0 计。每视频详细结果、失败解释与复现命令见 [实验与验证记录](D:/program/SightGuard/docs/path_relevance_validation.md)。

## 性能与自动化检查

- SegFormer 模型加载约 144.9 ms；排除每段首帧热身后的 232 次推理平均 17.23 ms，p95 为 18.96 ms。240 次推理的 p95 为 19.19 ms。
- 每视频采样帧处理均值范围为 42.6–67.3 ms，八个视频均值再平均约 54.0 ms。该计时从采样帧已解码后开始，包含后处理、对象关联、JSONL 和叠加图写入，不代表端到端实时速度。
- PyTorch CUDA peak allocated 为 196.11 MiB；这不是整卡或整个进程的总显存占用。批次墙钟约 19.0 秒，也不据此宣称实时 FPS。
- Ruff lint 检查通过，Ruff format 检查确认 169 个文件已格式化；完整 pytest 为 371 passed、6 skipped。Path Relevance 定向测试为 17 passed；独立 V05 本地 CUDA 模型集成测试为 1 passed，另有 2 条非致命 pytest 插件重写警告。

## 限制与后续工作

当前视觉标注仍为 `pending_human_annotation`，8 个任务的人类属性标签均为空。现有结果只说明模型与检测框在这些样本中的图像区域交叠，不足以判断真实地面边界、目标是否占道、误报或漏报。因此不报告 Accuracy、Precision、Recall、F1、误报下降或安全收益。

V04 的 30 帧中有 27 帧场景质量为 limited，108 个对象观察中 100 个为 UNKNOWN；V05 的宽地面区域可能合并道路与人行道，V03 的检测框交叠可能受遮挡和框高度影响。系统没有相机标定、姿态/旋转估计、度量深度、地面平面恢复或可靠动态遮挡处理，也没有评估可通行宽度或碰撞时间。

建议下一阶段先独立盲标候选地面边界和对象占用关系，再按场景质量、相机运动、遮挡及类别分层评估；只有完成标注和验证后，才讨论是否将某种研究信号接入正式提醒策略。

## 交付位置

- 代码及设计说明已包含在提交 `3902563c458832839bbf1d143612a934ecb53c34`（`experiment: integrate path relevance perception`）。
- 本报告与详细验证文档位于 `docs/`。
- v04 JSONL、视频叠加图、模型权重和原始视频受 `.gitignore` 管理，不随 Git 提交发布；权重哈希、输入标识和运行配置可由批次元数据追溯。
