> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Phase 2 实际验证记录

验证日期：2026-10-09（Asia/Shanghai），Windows、本地 uv Python 3.12.15。
本阶段没有增加运行依赖；版本更新为 SightGuard 0.2.0，uv.lock 同步更新项目版本，视觉依赖保持原锁定版本。
AGENTS.md 未改动，Phase 1 schema 1.0 保留，Phase 2 输出采用 schema 2.0。

## 实际运行结果

开发前 `uv run pytest -q`：24 passed、2 skipped，与 Phase 1 的默认无模型运行结果一致。

完成后实际运行：

```powershell
uv lock --offline
uv sync --locked --offline
uv run ruff check .
uv run ruff format --check .
uv run sightguard --help
uv run sightguard assess --help
uv build
uv run pytest -q
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q
Remove-Item Env:SIGHTGUARD_TEST_MODEL
```

依赖同步成功，Ruff 检查及格式检查通过，analyze/replay/assess 命令均保留且可启动；
0.2.0 wheel 与 sdist 构建成功。
默认套件：**83 passed、2 skipped**（2.13 秒）；跳过的两项仍是 Phase 1 的显式 opt-in 本地模型集成测试。
随后设置已有 YOLO26n 权重路径的完整套件：**85 passed**（6.89 秒），无跳过，包括 Phase 1 真实 CPU
检测跟踪及所有 Phase 2 测试。Phase 2 的 59 项测试不需要权重、网络或 GPU。
这些时间是单次 pytest 报告，不是算法性能基准。没有删改、跳过或弱化原有 Phase 1 测试。

开发中的 Ruff 行宽/导入顺序问题在重新格式化后修正。审查发现亚 10ms 帧间隔会使运动特征持续
不可用；已增加独立累积运动基线并通过 5ms 帧间隔测试。全文哈希原本用于 stream_id，追加未来帧
会改变前缀标识；已改为稳定路径/显式视频标识，全文哈希仅用于审计，并通过共享前缀逐条输出测试。

## 实际离线分析

```powershell
uv run sightguard assess --tracks tests/fixtures/phase2/static_occlusion.jsonl --config configs/risk.yaml --stream-id synthetic-01 --output-dir outputs/phase2_fixture_verified
uv run sightguard assess --tracks outputs/tracking_verified/tracks.jsonl --config configs/risk.yaml --output-dir outputs/tracking_verified/risk_phase2
uv run sightguard assess --tracks outputs/demo/tracks.jsonl --config configs/risk.yaml --output-dir outputs/demo/risk_phase2
```

上述三个实际命令全部成功，四种要求的输出均生成。感知模型没有重跑，原始 tracks.jsonl 不改写。
输出已存在时拒绝覆盖；复现请换新的输出目录。

合成静止/遮挡回归输入：5 帧、1 个风险事件、1 条首次提醒。
event_id=E000001：0ms 创建 CANDIDATE，250ms 激活 ACTIVE 并发送 A000001（HIGH/first），
500ms LOST，650ms 通过几何恢复 ACTIVE，原始 ID 从 7 换为 9；EOF 保持 ACTIVE 并记录 censored。
手工指定预期存于 static_occlusion.expected.json，测试同时检查具体状态、分数、时间、关联历史、
提醒字段、所有业务记录的版本号以及重复导出逐字节相等。两次运行元数据的审计时间允许不同。

Phase 1 的真实模型感知回放：10 帧、3 条提醒，源数据来自重复静态 bus.jpg 图像。
这只是实际感知数据能接入风险引擎的集成验证，没有独立风险标签，不能据此判断误报/漏报或道路效果。
空检测合成视频回放：20 帧、0 条提醒，不把空检测解释为安全结论。

## 自动化测试覆盖

- 空间归一化、梯形关联、横向经过、实际不规则 dt 下的尺度与中心变化、静止运动证据。
- 第一观察/缺 ID/短时间基线/重复时间戳等运动 unknown；高帧率时间累计。
- R/Q 分离、缺运动权重重分配、可选因果 EMA、阈值附近抖动与时间迟滞。
- 候选确认、静止不重复提醒、中到高升级独立间隔、稳定多目标排序与发送名额。
- 遮挡 LOST/恢复、ID 切换与几何恢复、两目标一对一、歧义拒绝、过期 ID 复用新事件。
- 无效框 partial 帧、候选确认重置、失败帧不使已知目标消失、持续受限的独立健康提示。
- 低帧率配置、时间倒退/缺失/乱序输入、跨视频状态隔离、EOF 截尾、未来帧不改变共享前缀。
- 配置无效范围/未知键、JSONL 导出、失败元数据、源感知失败信息、原输入不修改、输出拒绝覆盖。
- 独立子进程禁止导入 torch/ultralytics/cv2/numpy/gradio，assess 仍能完成。
- Phase 1 全部原有测试，包括实际 CPU 模型集成。

## 未验证与已知限制

未运行 GPU、Linux/macOS、真实动态道路数据、实际提醒硬件或真实用户测试。没有独立标注集、
重要风险召回率、提醒精确率、重复率/无效率、延迟或安全性能结论。
长序列资源开销没有性能基准。当前结构化结果之外不生成风险叠加视频（任务允许保留此扩展）。
已知图像运动、ID 交换/歧义、低帧率、缺运动降级评分和提示优先级等失效场景见 phase2.md。
Phase 3 的策略比较和标注评价平台尚未开始开发。
