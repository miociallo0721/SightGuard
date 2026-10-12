# 文档索引

优先从下面的当前说明进入。带日期、Phase、v01 等的报告保留当时的版本和测试边界，不代表最新代码状态，也不应被后续安装或实验结果覆盖。

## 使用与产品状态

- [项目首页](../README.md)：用户能力、当前限制和入口。
- [Android 使用与开发](../android/README.md)：自动感知、模型资产、构建与 ADB 验证。
- [PC 使用](pc-guide.md)：本地演示、真实模型和无模型重放。
- [当前能力与证据](product/status.md)：稳定版、实验版及验证等级。
- [尚未解决的问题](product/known-issues.md)：发布阻塞、算法失败和工程欠账。
- [版本、交付与回退](product/releases.md)：包与源码来源、发布前要求。
- [2026-10-12 公开数据评测](product/dataset-evaluation-20261012.md)：Metric3D 对照、NYU 语义真值、性能与失败；仅 PC。

## 开发与贡献

- [当前架构](architecture.md)：PC / Android 链路、研究隔离和时间契约。
- [PC / Android 能力差异](android_capability_matrix.md)：参数、跟踪、音频与研究模块边界。
- [贡献要求](../CONTRIBUTING.md)、[隐私与问题报告](../SECURITY.md)、[变更记录](../CHANGELOG.md)。
- [数据说明](../datasets/README.md)与[真实数据来源](real_world_benchmark_v01.md)：许可、独立标注和数据划分。

## Android 实现与历史验收

- [0.4 无障碍环境感知](../android/docs/accessible_awareness_v04.md)：实际模型输出、失败图片、声音、AVD 和 nova12 验收范围。
- [0.3 墙面事件](../android/docs/wall_awareness_v03.md)：墙面语义到声音链路的修复。
- [0.3 空间感知](../android/docs/spatial_perception_v03.md)及[验证记录](../android/docs/spatial_perception_validation.md)。
- [0.2 PC 对照交付](android_pc_parity_delivery.md)、[初始 MVP](android_mvp_delivery.md)：仅对应旧版本。
- [米制与局部候选研究](../android/docs/metric_corridor_research.md)：独立实验分支、复现及失败。

## PC 专项与研究记录

- [离线评分与事件](phase2.md)、[独立评测方法](phase3_methodology.md)、[指标定义](metrics_definition.md)。
- [标注工具](annotation_studio.md)、[真实数据开发集](real_world_benchmark_v01.md)。
- [实时摄像头](live.md)、[空间音频](spatial_audio_architecture.md)。
- [Path Relevance](path_relevance_v1.md)、[相对深度](depth_integration_v1.md)、[Grid](spatial_grid_fusion_v1.md)、[Corridor](corridor_intelligence_v1.md)、[运动质量](motion_quality_v1.md)。
- [Hybrid 架构](hybrid_intelligence_architecture.md)、[实际验证](hybrid_intelligence_validation.md)：辅助模型均为 Shadow。
- [2026-10-11 架构快照](architecture_audit_2026-10-11.md)与[当时的迁移计划](architecture_consolidation_plan.md)：保留历史判断，当前状态以本索引中的产品文档为准。

## 仓库整理原则

不移动冻结 benchmark 的输入、配置或历史原始结果。研究脚本和报告保留原路径，避免破坏复现命令；首页只保留用户和开发者常用入口。

Git 忽略的 `models/`、`outputs/`、`android/deliveries/` 是本地资产区，不是远程仓库中可保证存在的文件。交付资料应绑定源码 SHA、模型 / 输入哈希、运行环境和验证等级，新增结果写入独立目录。
