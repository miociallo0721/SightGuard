# SightGuard — Spatial Grid Fusion v1 阶段任务报告

日期：2026-10-11。

本阶段完成了 YOLO / BoT-SORT、SegFormer 地面语义和 Depth Anything V2
相对深度的独立离线空间证据融合，完成真实数据回放、四组同源消融、可视化与兼容性验证。
Spatial Grid 是图像空间证据表示，不是米制三维占用地图，不能作为安全通行或碰撞预测结论。

## 交付与版本

- 仓库：`miociallo0721/SightGuard`。
- 交付分支：`codex/spatial-grid-fusion-v1`。
- 实现提交：`5e7e735b4e437c60bcfe61c718fd3ccd7627f89e`，
  `experiment: integrate spatial evidence grid`。
- 本报告另作文档提交，与实现通过同一远端分支交付；不合并默认分支。
- 基线提交：`5291529998301f4f7a2162898fbc0d9f43ae2baf`，包含 Depth 实现
  `c45f30057bf659a5f7a0ccccefb61de5a84cdc47`。
- 模型、视频、NPZ、生成图片与实验输出保留在本地 Git 忽略目录，未纳入提交。
- 预先存在的未跟踪架构报告未修改、未纳入本次提交；AGENTS.md、uv.lock 和全局配置不变。

## 实现结果

核心代码位于 `src/sightguard/scene/spatial_grid/`，包含配置、严格帧身份、
来源契约、单元证据、对象关系、无状态融合与保存数据适配器。
核心算法没有模型、风险、决策、UI 或音频依赖。

新增入口：

- `scripts/prepare_spatial_semantics.py`：复用原 SegFormer 适配器，保存原生语义数组。
- `scripts/run_spatial_grid.py`：复用真实保存结果，执行融合与 A/B/C/D。
- `scripts/validate_spatial_grid.py`：冻结文件校验、帧对齐、消融一致性及独立风险重放。

默认 15×15 网格可配置，源图半开像素坐标完整覆盖非整除尺寸。
每格分别保存地面支持、候选组件联系、检测框覆盖、相对深度统计、质量、
冲突与未知原因，以及文件引用和哈希。
状态为 GROUND_EVIDENCE、OCCUPANCY_EVIDENCE、MIXED、UNKNOWN；没有 FREE 或安全概率。

语义 UNKNOWN 不被有效深度补成地面。负深度、NaN/Inf 不参与数值统计，
原始深度数组不修改；保存矩阵读取还遵守既有完整性校验。
检测框仅提供占用候选，多个目标分别保留，不创建 event_id 或改变跟踪身份。

## 真实输入与验证结果

UrbanEgo 八段开发视频来自同一 parent source。每段 30 个采样帧，
合计 240 帧，1280×720，frame_index=0,10,…,290，timestamp_ms=0,500,…,14500。

复用 `outputs/real_world/v01/` 跟踪数据与 `outputs/depth_integration/v03/` 深度矩阵。
Path Relevance v04 没有保存原始 mask，因而重新执行同模型、同预处理、同采样帧的
真实 SegFormer 推理，保存为独立的 `outputs/spatial_semantics/v01/`。
240 次模型推理成功，其中 238 帧有候选区域、2 帧 scene UNKNOWN；
scene status 与原 v04 对应帧相同，但不声称新数组与未保存的旧掩码逐像素一致。

最终融合目录为 `outputs/spatial_grid/v03/`；验证记录为
`outputs/spatial_grid_validation/v02/`。v01/v02 探索批次保留，未覆盖已有结果。
最终 metadata 的 source SHA 与实现源码一致，v02/v03 的 240 帧结果完全相同。

完整融合得到 54,000 个单元、1,123 个对象关系：

- GROUND_EVIDENCE：17,429。
- OCCUPANCY_EVIDENCE：10,147。
- MIXED：3,386。
- UNKNOWN：23,038，占 42.663%。
- 所有单元及对象都有可计算深度摘要；1,121 个对象有可计算语义比例。
- 7,924 单元包含冲突诊断；42,035 单元仍含未知或受限原因。

A 仅语义：20,815 地面单元，UNKNOWN 61.454%。
B 仅深度：54,000 单元数值有效，但状态全部 UNKNOWN。
C 检测与语义：状态分布与 D 完全相同。
D 三源：增加相对近远与深度变化证据，没有强行减少 C 的 UNKNOWN。
这些是工程覆盖和可解释性指标，不是准确率或安全性改善。

## 困难场景

已查看 RGB 与六面板真实对照，保留 V03/V04/V05/V07/V08 的不确定样本：

- V03 frame 200：大车、远车和遮挡导致框内背景混合，隐藏地面不能由深度恢复。
- V04 frame 200：215 格地面语义不足，225 格仍有深度；近墙和窄人行区域没有净空保证。
- V05 frame 290：8 个椅子框内地面支持比例约 0.351–1.000，存在 bbox 扩大占用的候选问题。
- V07 frame 100：多对象、遮阳棚、桌椅与招牌混合，29 格 MIXED；近深度不是地面阻断证明。
- V08 frame 0：低台阶/柱体不能凭深度结构成为可靠占用判断。
- V08 frame 290：俯视铺装地面时语义场景 UNKNOWN，225 格深度有效但仍全部 UNKNOWN。

上述视觉检查不是独立空间占用真值，未计算 Accuracy、F1、误报/漏报或规划成功率。

## 性能与检查

Intel i7-12650H、RTX 2050 Laptop 4GB，Windows 11。以下为最终批次实测 P50/P95：

- CPU 总融合：127.84 / 155.34 ms。
- 含 RGB 解码、输入校验、融合、序列化和可视化的完整采样帧：363.02 / 535.78 ms。
- Depth NPZ 校验/读取/解压：30.15 / 35.48 ms；本轮没有重跑 Depth 模型。
- 批次墙钟：119.20 秒，包含补充消融；融合进程峰值工作集 207.53 MiB。
- 单独 SegFormer GPU 适配器推理：21.83 / 25.10 ms；peak allocated 196.11 MiB，
  peak reserved 306.00 MiB。模型重推理与 CPU 融合分进程运行。

已有服务和探索批次未被停止，性能不是独占机器微基准；未验证完整实时系统目标帧率。
metadata 的 ablation_D 是选择已计算结果的耗时，D 的融合性能应读取 fusion_total。

实现阶段检查结果：

- 默认 pytest：457 passed、16 skipped，29.56 秒；新增纯算法测试 54 项。
- 真实保存输入集成：6 passed；实际 SegFormer GPU 推理集成：1 passed。
- Ruff check 通过；最终 format --check：202 files already formatted。
- 890 个冻结文件 SHA 保持不变。
- 八段 2,400 帧独立风险重放：risk_frames/events/alerts 与原结果逐行一致。
- 原有 257 条正式提醒保持不变，各段 33/55/34/20/36/25/48/6；没有音频调用。

本报告制作仅复核既有证据、补充文档和远端交付，没有再次运行模型或宣称新增实验结果。

## 保留限制与阶段边界

正常依赖同步曾因运行中 sightguard.exe 文件锁报 os error 32；未强制停止服务，
无文件锁条件下的正常同步成功仍未验证。
浏览器工具拒绝本地 file:// 页面；六面板 JPG 已查看，Node DOM 模拟已验证交互逻辑，
但真实浏览器渲染尚未验证，不能把模拟当浏览器 E2E。

最大空间判断局限是单目相对尺度、未知相机姿态、语义分辨率、遮挡和 bbox 背景，
加上缺少独立空间标注。三源推理成功不代表物理空间判断可靠。
本阶段没有实现 Corridor Planner、路径规划、动态引导、音频调度或移动端。

下一阶段可复用 FrameKey、逐格 evidence/quality/reasons、组件联系、ObjectGridRelation
和原始数组引用；应先建立独立标注、姿态和观测有效性，再评价候选走廊。

## 资料与复现

- [设计、固定 Roana 来源及数据契约](spatial_grid_fusion_v1.md)。
- [完整实验验证、各视频统计、模型哈希和复现命令](spatial_grid_validation.md)。
- 本地交互入口：`outputs/spatial_grid/v03/index.html`。
- 本地原始核验：`outputs/spatial_grid_validation/v02/validation.json`、
  `manual_review.json`、`result_parity.json`、`checks.json`。

Roana 仅参考固定版本的网格和框覆盖思想，没有复制其 AGPL 实现、测试或 fixture。
后续重放必须选择新的不存在输出目录，不覆盖冻结数据。
