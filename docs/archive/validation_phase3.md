> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Phase 3 实际验证记录

日期：2026-10-09（Asia/Shanghai）；Windows PowerShell；uv Python 3.12.15；SightGuard 0.3.0。
Phase 1/2 的 85 项测试保留，新增 Phase 3 测试 50 项。AGENTS.md 未修改。
增加 Matplotlib 为直接依赖，离线更新 uv.lock；未开发 Phase 4。

## 实际执行与结果

```powershell
uv lock --offline
uv sync --locked --offline
uv run ruff check .
uv run ruff format --check .
uv run sightguard --help
uv run sightguard benchmark --help
uv run pytest -q
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q
uv build
```

- 默认 pytest：133 passed，2 skipped；两项跳过是原有、显式启用的真实模型集成验证。
- 启用本地权重后完整 pytest：135 passed，无跳过，7.67 秒。测试没有联网下载权重或依赖 GPU。
- 真实集成验证使用 CPU；一项验证合成 MP4 的 YOLO 处理，另一项使用 Ultralytics
  随包 bus.jpg 的重复帧验证 BoT-SORT ID 及视频输出。这是实际模型推理回归，不是真实动态道路风险评价。
- Ruff check 与 format --check 通过。
- uv build 成功生成 sightguard-0.3.0.tar.gz 与 sightguard-0.3.0-py3-none-any.whl。

Phase 3 覆盖机会匹配、全漏检、重复、升级、早到/迟到、截止边界、同类重叠及竞争、
空真值、不确定区间、空/无效数据集、不同 FPS/不规则视频时间、因果前缀和状态隔离、
输入共享/评分一致、split 泄漏、真实来源失败、导入回滚、协议改动拒绝与执行失败状态。
真实导入的成功道路资料尚不可验证；来源/导入异常用临时文件和模拟视频探测测试，不假装人工数据。

## 执行合成实验

```powershell
uv run python scripts/generate_phase3_fixtures.py
uv run sightguard validate-labels --dataset datasets/synthetic/manifest.csv
uv run sightguard freeze-protocol --dataset datasets/synthetic/manifest.csv --config configs/benchmark.yaml --risk-config configs/risk.yaml --output outputs/phase3_final_protocol.json
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split test --config configs/benchmark.yaml --risk-config configs/risk.yaml --protocol outputs/phase3_final_protocol.json --output-dir outputs/phase3_final
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split test --config configs/benchmark.yaml --risk-config configs/risk.yaml --protocol outputs/phase3_final_protocol.json --output-dir outputs/phase3_repeat
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split development --config configs/benchmark.yaml --risk-config configs/risk.yaml --output-dir outputs/phase3_development
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split test --config configs/benchmark.yaml --risk-config configs/risk.yaml --protocol outputs/phase3_final_protocol.json --strategies C --output-dir outputs/phase3_c_only
```

上述命令实际成功；manifest 共 5 段 synthetic：开发集 1 段，测试集 4 段。
测试有效时长 13 秒，3 个重要事件、4 个提醒机会，无排除时段。场景为静止目标、
尺度增长升级、完全感知漏检和无风险情境；窗口及预期计数在运行策略前声明于固定用例。
合成原始视频路径为空，video_sha256 是预定义场景指纹，不能理解为真实视频哈希。

实际计数（TP/FN 是机会单位，FP 包含重复提醒）：

- A：TP=2，FP=34，FN=2；重复 22，其他无支持 12，总提醒 36。
- B：TP=2，FP=2，FN=2；重复 1，其他无支持 1，总提醒 4。
- C：TP=3，FP=1，FN=1；重复 0，其他无支持 1，总提醒 4。

三者重要事件召回及按时提醒率均为 2/3；机会召回分别 1/2、1/2、3/4。
首次响应延迟均值分别 0、0、250 ms，仅统计两个成功首次提醒，另一个重要事件显式计为遗漏。
重复率分别 22/36、1/4、0/4。无效提醒/视频小时分别 9415.3846、553.8462、276.9231；
总提醒/视频分钟分别 166.1538、18.4615、18.4615。如此大的归一化值来自仅 13 秒的
刻意构造测试，不能外推真实一小时体验。

C 在此用例内成功提醒升级，B 的固定冷却错过升级；C 相对 B 总提醒数没有下降。
C 也在非风险情境发出一次无支持提醒。三个策略都无法弥补完整感知漏检。
结果只验证实现，不支持“已证明道路效果优于基线”或统计显著性结论。

## 确定性与产物

使用 Get-FileHash SHA256 对两次运行比较，以下 13 个文件全部字节一致：
metrics.json、comparison.csv、per_video.csv、per_scene.csv、per_severity.csv、per_event_type.csv、
event_results.csv、alerts.jsonl、matches.jsonl、error_cases.csv、risk_frames.jsonl、events.jsonl、report.md。
审计时间和离线墙钟耗时会随运行变化，不要求 run_metadata.json 或图表 SVG 随机 ID 字节一致。
测试同时验证生成器在两个临时目录生成相同内容，且等于保留的合成用例。

最终产物在 outputs/phase3_final；包含 comparison PNG/SVG 和 4 段时间线 PNG/SVG，共 10 个图文件。
已打开并目视检查比较图与升级时间线，数据类型标记可见；时间线绿色为匹配、橙色为重复、
红色为无支持、灰色为忽略、紫色为诊断，黑色叉号为遗漏机会截止点。
元数据记录三个策略相同的 tracks SHA、标签/数据划分、配置、源码、依赖锁、版本及冻结协议哈希。
输出只能在 status=completed 且 artifacts_complete=true 时使用；失败原始流为 .partial。

## 未验证与后续工作

没有合法来源、独立人工复核的真实动态风险测试集；也没有使用者、安全性、端侧实时耗时或
足够独立拍摄组的统计置信区间验证。当前 event_type 是真值情境描述，不是算法预测语义。

真实实验须采集/引入原视频，按原始拍摄组划分，运行 Phase 1 后 import-video，盲化独立
标注和复核，seal-labels/validate-labels；仅在 development 调参，然后冻结协议并运行 test。
保留相机运动、短暂遮挡、图像质量差、不确定事件和无风险视频，不能只保留模型检出的目标。
文件哈希能检测冻结后变化，不能证明人类标注确实盲化；审核记录和独立标注流程仍必要。
