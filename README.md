# SightGuard（视界）

## 当前运行架构（2026-10-11）

正式结构化提醒主链路为 **YOLO26n / BoT-SORT → FrameRecord → AssessmentEngine
（风险特征、评分、事件与感知状态）→ AlertScheduler → AlertEvent**。
实时摄像头和离线 tracks 重放复用同一核心；空间音频需要显式启用，再由
RiskCueAdapter / CueScheduler / AudioWorker 提交到 PC 的 OpenAL Soft。
存在 AlertEvent 不代表声音已播放或用户已听到。

- `sightguard ui`：默认 FastAPI 产品演示页；`sightguard ui --research`：Gradio Research Console。
  当前 CLI 总帮助仍将 `ui` 描述为 Research Console，实际路由以上述开关为准。
- `sightguard live`：OpenCV 摄像头规则演示；`analyze`：仅 MP4 感知；`assess`：已有 tracks
  的正式规则评估；`replay`：仅校验与摘要。产品上传任务顺序执行 analyze 和 assess。
- SegFormer / Path Relevance v1：独立 `path-relevance` 离线 Shadow，未进入默认实时摄像头；
  Research Console 只浏览其产物。最新权威研究批次是 `outputs/path_relevance/v04/`，
  CLI 默认目录 v01 不表示最新结果。
- Event Association v2：`allow_class_switch_for_track_id` 默认 false，只有显式实验配置启用。
  Laya Structured / Vision / Hybrid 均为按需加载的 Shadow，不控制正式提醒或 HRTF。
- 产品 MP4 的 DemoGuidance 方向建议、语音试听及手动 NavigationCue 是演示/模拟来源，
  不等于正式 AlertEvent，更不证明通行安全。

架构整理阶段只修改文档。随后 Depth Integration v1 新增独立的本地相对深度研究入口，
不改变上述默认主链路；尚未实现统一 SceneEvidence、Corridor Grid、度量深度或可靠走廊决策。
接手开发先阅读
[实际架构审计](docs/architecture_audit_2026-10-11.md)、
[Roana 源码对照](docs/roana_architecture_comparison.md)和
[契约与分阶段迁移计划](docs/architecture_consolidation_plan.md)。

## Depth Integration v1

Depth Anything V2 Small 的真实 CPU/CUDA 推理、原生预测和原图对齐 float32 NPZ、
帧级 JSONL、近远语义及 UNKNOWN 检查均独立于提醒。UrbanEgo 八段各 30 个采样帧；
当前完整批次在 `outputs/depth_integration/v03/`。较大值表示画面内相对更近，没有米制距离。
入口为 `scripts/run_depth.py`，不加载 YOLO/SegFormer/Laya，不发音频。
详见[模型与数据契约](docs/depth_integration_v1.md)和[真实验证记录](docs/depth_integration_validation.md)。

## Annotation Studio

`uv run sightguard ui --research --port 7861` → Annotation Studio：原视频 PTS 逐帧查看、
可点击人工时间轴、事件/提醒机会/排除区间编辑、Schema 3.0 保存及显式二次审核。
不会自动生成真值；八段真实数据仍待人工标注。
使用步骤见[标注工具文档](docs/annotation_studio.md)，实测记录见[验证文档](docs/annotation_studio_validation.md)。

## Real-World Benchmark v0.1

已准备 UrbanEgo 同一次步行记录的 8 段开发视频，完成本地感知、风险日志和
Phase 3 draft 导入。人工真值尚未完成，不提供真实监督指标；数据仅保存在 D 盘。
来源、许可、复现命令及独立标注步骤见[数据集阶段文档](docs/real_world_benchmark_v01.md)。

## Hybrid Intelligence（影子研究）

已有产品网页的“实时体验”新增 Risk Intelligence 面板，支持 Rule / Laya / Vision /
Hybrid 四种研究模式。**Actual Alert Controller 始终是 Rule Engine**；辅助判断与实验性融合
不会控制实际提醒或空间音频。先启动摄像头，再选模式并主动点击“加载辅助模型”；
切换模式会卸载旧辅助模型，随后可加载新模型，摄像头、YOLO 和现有风险事件保持连续。

本机历史记录已验证 Laya CPU INT8 ONNX、Vision CPU FP32，以及隔离环境的 Vision CUDA
混合精度前向；Structured CUDA 仍有核心量化运算回退 CPU。它们是研究验证，不是风险准确率验证。
`configs/hybrid.yaml` 配置本地权重、隔离解释器、设备、限频、超时、有效期和融合策略。
默认不下载、不启动辅助模型；当前配置 Structured 为 CPU INT8 ONNX，Vision 为
`device: auto` / FP32（实际设备由隔离解释器能力决定）。两套上游包同名 `laya`，
使用独立环境与单一辅助子进程。真实 GPU 研究记录见 [GPU 可行性记录](docs/hybrid_gpu_feasibility.md)。

```powershell
# 从项目根目录执行；本次所有临时与缓存文件均使用 D 盘
New-Item -ItemType Directory -Force models/hybrid/tmp,models/hybrid/cache | Out-Null
$env:TEMP = "$PWD/models/hybrid/tmp"
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:UV_CACHE_DIR = "$PWD/models/hybrid/cache/uv"
$env:HF_HOME = "$PWD/models/hybrid/cache/hf"
$env:TORCH_HOME = "$PWD/models/hybrid/cache/torch"
uv run sightguard ui --port 7860
# 独立真实模型测试：必须已准备本地权重，不会自动下载
$env:SIGHTGUARD_TEST_HYBRID = '1'
uv run pytest tests/test_hybrid_integration.py -q --basetemp models/hybrid/tmp/pytest-real
uv run python scripts/verify_hybrid_live.py --camera 1
```

运行记录保存于新的 `outputs/hybrid/session-*`；包含原始输出、同源帧比较、实际规则提醒、
配置/权重哈希及性能统计。规则日志仅记录语义任务采样帧，不是完整视频；Vision 原图默认
只存像素哈希，需精确离线重放图像时显式设 `retain_images: true`，注意磁盘占用与隐私。
候选得分未校准，模型间一致率不是准确率。Vision 权重 CC-BY-NC-SA-4.0，发布需审查许可。
环境重建、固定来源和限制见[架构文档](docs/hybrid_intelligence_architecture.md)，
真实验证数据与兼容性见[验证记录](docs/hybrid_intelligence_validation.md)。

## 智能辅助眼镜展示网页

面向高中科创展示的产品介绍与视频模型演示现作为默认网页入口：

```powershell
uv sync --locked
uv run sightguard ui --port 7860 --open-browser
# 原 Gradio 研究控制台：
uv run sightguard ui --research --port 7861
```

打开 http://127.0.0.1:7860，上传 MP4 后点击“开始识别”。模型检测结果、方向演示建议、
语音提示文字与单侧马达节奏跟随视频帧同步。网页提供点击语音试听，硬件马达尚未连接。
方向规则仅用于演示，不是经过验证的导航；未知/空检测不会被视为可安全前进。
页面和规则、配置、输出格式及运行方式见 [展示网页说明](docs/product_showcase.md)。

面向视障辅助预警研究的本地、离线基础工程。当前完成 Phase 1 视觉感知、Phase 2
启发式动态风险评估、事件管理及结构化提醒：MP4 → YOLO26n → BoT-SORT → 感知 JSONL →
可解释风险特征/评分 → 风险事件 → AlertEvent，以及 Phase 3 独立标注与策略实验基础设施、
Phase 4 Gradio Research Console（研究控制台）。
感知结果支持独立重放，风险分析和策略对照不重新运行模型。

系统是盲杖、导盲犬的补充，不提供完整导航、路径规划或可靠碰撞预测。
尚未开展真实使用者验证、带独立标注的真实视频实验或真实道路安全性验证，不应据此做安全决策。

## Windows 安装

在 PowerShell 中进入项目目录，先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后：

```powershell
uv python install 3.12
uv sync --locked
uv run sightguard --help
```

`uv.lock` 固定实际依赖版本，`.python-version` 固定 Python 3.12 系列。
OpenCV 的 MP4 编码器取决于本机 wheel/编解码支持。Linux/macOS 同样使用上述 uv 命令。
默认使用 CPU，GPU 用户需要自行安装适配驱动的 PyTorch 并记录环境变化。

## 本地模型与视频分析

默认分析不会自动下载权重。首次联网准备模型（Ultralytics 官方模型资源），之后可离线运行：

```powershell
uv run python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt')"
uv run python scripts/generate_demo.py
uv run sightguard analyze --input examples/demo.mp4 --output-dir outputs/demo --config configs/default.yaml
uv run sightguard replay --input outputs/demo/tracks.jsonl
```

可用 `--model models/your-model.pt`、`--device cpu` 或 `--device 0` 覆盖配置。
配置中的 `image_size`、`confidence`、`class_ids` 支持调整；默认 COCO person=0、bicycle=1、
motorcycle=3；例如 `[0, 1, 2, 3, 5, 7]` 加入车辆，`null` 处理所有模型类别。
默认置信度 0.25 会过滤更低置信候选，包括 BoT-SORT 二阶段匹配可用的部分候选；
研究跟踪连续性时可将其设为 0.1，并在对照实验中固定该参数。
模型应兼容 Ultralytics detect/track，Phase 1 标准模型是 YOLO26n。
相对路径基于命令执行目录，而非配置文件目录。输出目录已有同名产物时拒绝覆盖。
真实视频由用户本地提供；合成矩形视频只能检验 I/O，不检验实际检测能力。

输出文件：

- `tracks.jsonl`：每个成功解码帧一行，包括空检测帧，零基帧编号、视频毫秒时间、尺寸、类别、
  置信度、xyxy 像素框及可空 ID。`schema_version="1.0"`。
- `annotated.mp4`：框、类别、ID、置信度及帧信息；`#?` 表示没有跟踪 ID。
- `run_metadata.json`：输入/模型/跟踪配置 SHA-256、软件版本、完整配置、硬件、解码信息、
  帧统计、时间来源计数、状态与异常。运行耗时只是处理记录，不是经过基准测试的性能结论。

`status="ok", detections=[]` 表示这一帧返回空结果；`status="processing_error"` 表示感知处理失败。
发生处理失败会写失败帧、保留部分输出并终止，避免继续使用不确定的跟踪状态。
输入解码失败没有有效图像，记录于运行元数据，不伪造成功帧。CLI 错误退出码为 1。
评估前必须检查元数据 `status` 和 JSONL 帧状态，不能将部分运行当作完整实验。

## 时间与跟踪限制

优先使用 OpenCV `CAP_PROP_POS_MSEC`，无效或不递增时采用
`max(frame_index*1000/fps, previous_timestamp+1000/fps)`；记录 `timestamp_source`。
FPS 缺失时拒绝运行。标注视频按原 FPS 编码、不保留音频；可变帧率视频的精确时间分析
应使用 JSONL，而标注视频是恒定帧率预览。OpenCV 若无法报告总帧数，读失败可能是结束或解码故障，
元数据明确标记 `eof_or_decode_failure_unknown`；不保证识别所有损坏视频。
BoT-SORT ID 只是当前视频会话中的估计关联标识，可能丢失、切换或重置，不代表真实世界稳定身份。
每帧都调用跟踪，包括空检测帧；各视频创建独立模型与跟踪器。

## 测试与检查

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

默认测试用模拟检测结果和本地生成视频，不需要模型、网络或 GPU。
单独真实推理验证需显式指定已准备的本地权重：

```powershell
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q -m integration
Remove-Item Env:SIGHTGUARD_TEST_MODEL
```

此集成测试只验证真实推理、BoT-SORT 与输出链路；真实目标追踪能力须另用有目标的视频验证。
实际本次验证见 [docs/validation.md](docs/validation.md)，架构和数据格式见
[docs/architecture.md](docs/architecture.md)。权重、视频、输出与虚拟环境不提交 Git。
Ultralytics 使用前应阅读其 [许可说明](https://www.ultralytics.com/license)。

## 后续阶段

Phase 3 和 Phase 4 的研究功能已实现。后续可研究性能优化、ONNX/Core ML 与 iOS 部署；
`assess` 仅导出结构化提醒，不播放音频；实时链路可显式启用 PC 空间音频，见下文 Spatial。
不提供经验证的导航或移动端功能。

## Phase 2：离线风险分析

无需下载模型或使用 GPU，直接复用感知 JSONL：

```powershell
uv run sightguard assess --tracks outputs/demo/tracks.jsonl --config configs/risk.yaml --output-dir outputs/demo/risk
```

仓库还提供独立于视觉模型的合成回归输入，首次即可运行：

```powershell
uv run sightguard assess --tracks tests/fixtures/phase2/static_occlusion.jsonl --config configs/risk.yaml --stream-id synthetic-01 --output-dir outputs/phase2_example
```

参数 `--stream-id` 可指定稳定的视频标识；默认使用输入绝对路径的哈希，追加未来帧不会改变
早期记录的标识。event_id/alert_id 为单次视频会话内的顺序编号，引用时同时保留 stream_id
和运行来源，不把它们作为全局真实目标身份。输出目录已有产物时拒绝覆盖。

输出均为 schema 2.0：`risk_frames.jsonl`（每帧特征、S/A/M/R/Q 与事件快照）、
`events.jsonl`（创建、更新、激活、丢失、恢复、结束/截尾及原始 ID 关联）、
`alerts.jsonl`（首次、升级、可选重复、感知受限提示）、`run_metadata.json`（输入与源码哈希、
有效配置、版本、统计与失败原因）。原始 tracks.jsonl 不改动。

默认 R=0.45S+0.35A+0.20M，Q 独立；缺少运动证据时重新归一化 S/A 权重，不填零。
中央梯形只近似画面关注区域；框尺度变化只表示图像变化，不表示距离、真实速度或碰撞时间。
风险等级工程初值为 0.35/0.65，带迟滞；普通候选需 250ms 和至少两次有效观察，恢复窗口 750ms。
默认不重复提醒同等级事件；可配置 3000ms 普通重复间隔，升级间隔独立为 1000ms。
持续感知失败或受限 500ms 后形成独立提示；失败不是空场景，也不会据此降低已知事件风险。
输入结束时未结束事件标记 `censored`，不声称障碍消失。

详细公式、完整参数、状态机与输出格式见 [docs/phase2.md](docs/phase2.md)，本阶段实际验证见
[docs/validation_phase2.md](docs/validation_phase2.md)。当前仅导出结构化风险与提醒，不额外绘制风险视频。

## Phase 3：独立标注与策略对照

Phase 3 核心不依赖 UI，不训练新模型。A 为固定类别/关注区逐帧触发；B 使用共享的 Phase 2
风险评分加全局固定冷却；C 直接调用现有 AssessmentEngine。三者复用相同感知数据，
只使用当前及历史视频时间。独立人工真值不使用检测/跟踪 ID 建立身份。

PowerShell 合成验证（明确是 synthetic，不代表真实道路效果）：

```powershell
uv run python scripts/generate_phase3_fixtures.py
uv run sightguard validate-labels --dataset datasets/synthetic/manifest.csv
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split development --config configs/benchmark.yaml --risk-config configs/risk.yaml --output-dir outputs/phase3_development
uv run sightguard freeze-protocol --dataset datasets/synthetic/manifest.csv --config configs/benchmark.yaml --risk-config configs/risk.yaml --output outputs/phase3_protocol.json
uv run sightguard benchmark --dataset datasets/synthetic/manifest.csv --split test --config configs/benchmark.yaml --risk-config configs/risk.yaml --protocol outputs/phase3_protocol.json --output-dir outputs/phase3_test
```

每次使用新的输出目录，冻结协议拒绝覆盖。源码、依赖锁、配置或数据变化后协议校验会失败；
应在开发集上完成修改，建立新的实验版本，不能通过重新冻结来掩盖测试集调参。
`--strategies A` / `--strategies B` / `--strategies C` 可单独执行；默认运行全部。
可在 YAML 中关闭 `generate_figures`。真实数据与合成数据分别报告，混合 manifest 用
`--data-kind real` 或 `--data-kind synthetic` 选择。校验仍会检查整个 manifest 的划分泄漏和文件证据。

产物包括 comparison.csv、metrics.json、per_video.csv、per_scene.csv、per_severity.csv、
alerts.jsonl、matches.jsonl、error_cases.csv、共享 risk_frames.jsonl、C 的 events.jsonl、
report.md、run_metadata.json，以及 Matplotlib PNG/SVG 比较图和每视频提醒时间线。
失败会保留 `status=failed` 和失败阶段；空提醒不等于输入缺失。CSV 未定义数值留空，JSON 为 null。

真实视频导入和人工标注流程见 [数据集说明](datasets/README.md)。
详细协议、数学定义、标注规范与本机实际验证分别见
[方法学](docs/phase3_methodology.md)、[指标](docs/metrics_definition.md)、
[标注规则](docs/annotation_guidelines.md)、[验证记录](docs/validation_phase3.md)。

另外导出 event_results.csv（包括不可评价事件和排除原因）以及 per_event_type.csv
（按人工情境分层的事件召回）。情境分层不能解释为算法实现了情境分类。
最终实际回归为 135 passed；默认不启用权重时为 133 passed、2 skipped。
合成测试结果及 13 个确定性文件的验证见 docs/validation_phase3.md。

## Phase 4：Research Console

在项目根目录的 PowerShell 中安装锁定依赖并启动：

```powershell
uv sync --locked
uv run sightguard ui --research --port 7860
```

打开 http://127.0.0.1:7860；可加 `--open-browser` 自动打开浏览器。仅绑定本机回环地址，
不创建公开分享链接，关闭 Gradio/Hugging Face 遥测。构建页面不加载模型；已有数据浏览无需权重。

控制台默认进入“产品概览”展示首页：深色背景、青绿色感知概念图和能力介绍，
可通过“开启实时体验”或“探索视频回放”进入实际功能。概念图为本地 SVG 绘制，
不代表实时检测或性能数据。导航为“产品概览 / 视频回放 / 策略对比 / 实验洞察 / 数据资料 / 实时体验”。
逐帧数据、原始指标与实时事件详情默认折叠，展开后仍可查看完整记录。

- **视觉分析**：选择 `outputs/` 内已有分析运行并加载，查看标注视频、实际统计、风险曲线、
  帧观测和事件生命周期。时间滑块选择请求时间之前最近的已记录帧；显示实际帧号/时间。
  生命周期下拉框选定记录后点“定位该记录”；“导出当前分析结果 ZIP”创建独立副本。
  上传 MP4 后选择 perception/risk 配置并主动开始分析，调用现有 YOLO26n/BoT-SORT 和 Phase 2。
  必须先准备 `models/` 内本地权重；配置通过 `configs/` 下的 YAML 管理。
- **策略对比**：选择已完成 Phase 3 benchmark 目录，按视频、A/B/C 和人工/模型事件筛选。
  共同时间轴展示提醒类型、匹配状态、人工提醒窗口和遗漏机会；指标直接读取已保存结果。
  展开感知重放预览，可选择 manifest、视频、配置和策略检查行为；此入口不加载真值、不报告准确性。
- **实验分析**：读取同一 `metrics.json` 以及分视频、场景、严重程度和错误案例 CSV。
  切换统计图、筛选策略与视频 ID。事件召回和提醒机会召回分别呈现；未定义数值显示 `N/A`。
- **数据集管理**：浏览 `datasets/` 中 manifest 的来源、划分、审核及路径状态，点击校验调用
  Phase 3 校验器。页面只读；导入/审核仍使用既有 CLI 与人工标注流程。

运行目录和 manifest 下拉列表来自本地真实文件。新分析可点“刷新任务目录”；其他目录在启动时扫描，
新增 benchmark/manifest 后重启控制台。没有数据的安装允许正常启动并显示空状态。
内置合成 manifest 可用；benchmark 须先按 Phase 3 流程生成，再由界面加载。
历史实验保留其原始源码/依赖证据；升级 UI 后不会改写冻结协议或重算历史指标。

界面模块：`ui/app.py`（页面）、`services.py`（受限读取/校验）、`jobs.py`（新分析/媒体缓存）、
`presentation.py` 与 `plots.py`（原值展示）。详细接口和验证记录见
[UI 架构](docs/ui_architecture.md)、[Phase 4 验证](docs/validation_phase4.md)。

### 使用限制

上传限 500 MiB，模型任务一次运行一个；界面会读取整份结构化结果，大型数据集仍需关注内存与磁盘。
允许的视频路径为项目内 `outputs/`、`datasets/` 或 `examples/`，配置/数据引用不得越界。
OpenCV 标注 MP4 经内置 FFmpeg 生成 H.264 浏览器缓存副本，原文件不变；首次转码可能耗时。
`outputs/console/` 保存新运行、上传及媒体/导出缓存，不自动清理。

播放器与时间滑块独立操作，未实现播放位置双向同步。图像帧与记录按时间定位；标注视频为恒定 FPS
无音频副本，VFR 原视频的播放时间不能替代 JSONL 时间戳。来源不明的视频显示 `unknown`，
不会因“是真实文件”就认定为经审核的真实道路实验。合成数据、重复静态图片视频仅验证工程流程。
浏览器检查覆盖桌面和笔记本布局及主要交互；触屏、其他浏览器和长视频操作仍需人工检查。

### 控制台测试

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
# 可选：仅使用已有本地权重，不下载模型
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q
Remove-Item Env:SIGHTGUARD_TEST_MODEL
# 已启动控制台且已有指定 benchmark/analysis 时，执行实际 HTTP 验证
uv run python scripts/validate_console.py --run-analysis --output outputs/console_validation/new-report.json
```

HTTP 验证脚本默认读取本机 `phase3_final` 和 `tracking_verified` 运行；其他机器可用
`--benchmark-run` / `--analysis-run` 指定已生成的目录。实际上传验证需要本机
`examples/tracking_fixture.mp4` 及模型权重；报告拒绝覆盖，不能把本机未提交的视频当作内置样例。

## Phase 5 P0–P2：实时 PC Demo

使用本地权重和 OpenCV 原生窗口，实时复用现有检测、跟踪与风险决策链路：

```powershell
uv run sightguard live --camera 0 --device auto --width 640 --height 480 --imgsz 640
# 本机 Xiaomi USB 摄像头为 DirectShow 编号 1：
uv run sightguard live --camera 1 --device auto
```

Q 退出，H 切换详细信息。容量为 1 的缓冲区保留最新待处理帧；结果绑定其原始画面，
过期状态显示 UNKNOWN。窗口显示实际评分、事件、提醒原因、风险更新 FPS 和处理耗时。
设备编号随机器和设备变化；使用 `--config`、`--risk-config`、`--live-config` 读取配置，
`--duration` 可指定自动退出秒数。启动方式、接口及本次验证见 [实时 Demo 文档](docs/live.md)。
本次仅为 P0–P2 工程原型，不是完整 Phase 5 发布或真实道路风险效果验证。

### 在网页中使用实时 Demo

```powershell
uv run sightguard ui --research --port 7860
```

打开 http://127.0.0.1:7860 的“实时体验”页；默认优先选择 Xiaomi 摄像头，
点击“开始实时体验”。查看实际风险画面、事件、提醒原因、风险更新 FPS、处理耗时和帧龄。
“结束体验”、切换到其他页面、刷新/关闭网页都会清理该页面的会话；
浏览器异常断连时由 10 秒心跳超时兜底释放。停止后可再次启动，全新跟踪和事件状态不会跨会话复用。
同一控制台只运行一个摄像头会话，其他页面不能抢占或读取该会话画面。
摄像头由本机 Python 读取，JPEG 通过内存更新，不创建摄像头视频或逐帧磁盘缓存。
原有 OpenCV `live` 命令继续可用。使用限制及验证见 [实时 Demo 文档](docs/live.md)。


## Spatial：PC 空间音频与模拟导航

音频默认关闭，声音由服务所在 PC 的 OpenAL Soft 输出，不通过浏览器播放器。
请先连接双耳耳机，在试听页面确认输出设备和实际 HRTF 状态。

```powershell
# 首次安装原生运行库（显式下载官方 Windows x64 发行包）
uv run python scripts/prepare_openal.py
# 独立试听页面
uv run sightguard spatial-demo --port 7862
# 研究控制台：Spatial Audio Lab 与实时体验页
uv run sightguard ui --research --port 7861
# Xiaomi 摄像头，本机 DirectShow 编号 1；其他机器需重新确认编号
uv run sightguard live --camera 1 --device cpu --spatial-audio
```

产品网页 `uv run sightguard ui --port 7860` 的“实时体验”区可直接选择摄像头、
开启空间音频、试听左/前/右方向，并查看实际 HRTF/设备/错误。
研究台实时页的高级设置也可开启音频并设置增益；运行中可静音。
模拟导航与该实时会话的风险提醒共享队列，支持播放、暂停、恢复和取消。
首次提醒、升级和感知受限只来自现有 AlertEvent；试听页手动风险音是测试音。
音频不可用会显示错误，视觉流程继续运行。未知方向使用中性正前方提示，
默认屏幕位置映射只是近似方向，不表示测得的世界方位、距离或碰撞概率。
同进程只允许一个音频会话；不要同时在独立 Lab 和其他服务进程中播放。

原生 DLL、下载包、试听记录均在 Git 忽略目录内。Windows x64 自动准备脚本
保留官方许可、版本与哈希；Linux/macOS 需要安装系统 OpenAL Soft 或配置
`SIGHTGUARD_OPENAL_LIBRARY` / `configs/spatial.yaml` 的 `library_path`。
详见 [空间音频架构](docs/spatial_audio_architecture.md)与
[验证记录](docs/spatial_audio_validation.md)。前后/高度定位、头部跟踪、实际导航及道路效果尚未验证。

本机 Xiaomi 黑屏/横纹问题已通过先设置尺寸与帧率、最后设置 MJPG 修复。
采集格式与请求帧率由 `configs/live.yaml` 配置；时间戳仍使用 monotonic，不按请求 FPS 推算。

Hybrid GPU 研究工具与实际限制见 [GPU 可行性记录](docs/hybrid_gpu_feasibility.md)。
2026-10-11 已完成 RTX 2050 上 Vision 混合精度真实推理；默认不加载辅助模型。
按需加载仍为 Shadow，设备以当前配置及隔离解释器能力为准，见本文开头的运行状态说明。
原风险任务仍全部 UNKNOWN，GPU 加速不等于风险判断有效；见 [UNKNOWN 调查](docs/laya_vision_unknown_investigation.md)。

提醒行为诊断：`uv run python scripts/diagnose_alert_behavior.py` 只读取现有真实运行日志，
输出逐视频统计、事件异常索引与原始评分轨迹，不重新推理或改变策略。
Research Console 的视频回放页提供事件筛选、原始曲线、实际 AlertEvent 标记与关键帧定位。
统计口径、证据和下一阶段建议见 [诊断报告](docs/alert_behavior_diagnosis.md)。

Spatial Grid Fusion v1 提供独立离线 Shadow：YOLO 跟踪、SegFormer 地面语义与
Depth 相对深度在同一原图坐标下形成可配置 15×15 证据网格，保留 UNKNOWN、MIXED 和溯源。
它不是米制占用地图或导航，不改变正式风险、提醒和音频。
运行 `uv run --no-sync python scripts/run_spatial_grid.py --output outputs/spatial_grid/new-run`，
输出目录必须不存在；首次缺少原始语义缓存时使用 `scripts/prepare_spatial_semantics.py`。
设计、上游参考与数据契约见 [实现说明](docs/spatial_grid_fusion_v1.md)，
240 帧真实融合、A/B/C/D、性能、限制与复现见 [验证报告](docs/spatial_grid_validation.md)。
