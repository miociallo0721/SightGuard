# SightGuard Real-World Benchmark v0.1

所有视频仅保存在本地 D 盘。`raw/` 为不可覆盖的来源归档/视频，`processed/` 为研究副本，
`manifests/` 为来源和处理记录，`annotations/` 为独立人工标注，`splits/` 为同源分组划分，
`reports/` 为质量检查和运行记录。原始视频、运行产物、关键帧及未复核标注默认不提交 Git。

来源核查见 `manifests/source_catalog.json`。下载、校验、时间映射与准备命令见
[阶段文档](../../docs/real_world_benchmark_v01.md)。Phase 3 的 schema 3.0 继续作为风险 Ground Truth；
来源 manifest 的 v0.1 不替代风险标注 schema。空 draft 不是无风险真值，只有人工 reviewed
标注可用于监督指标。首批全部为 development，不建立同源片段混入测试集的伪独立划分。
