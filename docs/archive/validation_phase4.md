> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Phase 4 本机验证记录

验证日期：2026-10-09。工作目录：`D:\program\SightGuard`，Windows PowerShell，Python 3.12.15。
SightGuard 0.4.0；Gradio 6.30.0、Plotly 6.9.0、Pandas 3.0.6、imageio-ffmpeg 0.6.0、
Ultralytics 8.4.174、PyTorch 2.14.1；实际解析版本以 `uv.lock` 为准。
Phase 1–3 有效测试全部保留，未修改 `AGENTS.md`。本机目录未初始化 Git，不能提供 Git diff。

## 实际命令与结果

```powershell
uv sync --locked --offline
uv run sightguard ui --help
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q
uv build
uv run sightguard ui --port 7860
```

- 锁定依赖离线同步成功；CLI 显示 `--port` / `--open-browser`。
- Ruff 检查与格式检查全部通过。
- 最终默认测试：**175 passed, 2 skipped in 9.01s**。跳过项为需主动指定本地权重的模型集成测试。
- 最终启用本地权重：**177 passed in 12.92s**；未要求 GPU，也未下载权重。
- 构建成功：`dist/sightguard-0.4.0.tar.gz` 和 `dist/sightguard-0.4.0-py3-none-any.whl`。
  使用 Python zipfile 实际确认 wheel 含 `sightguard/ui/styles/console.css` 和全部 UI 模块。
- 本机 `127.0.0.1:7860` 实际启动，启动 HTTP GET/HEAD 返回 200。未启用公开分享。

新增 42 项测试覆盖服务读取、异常类型/schema、空结果、路径越界/符号链接、来源标识、
缺失/变更真值、输入/风险结果哈希绑定、会话读取隔离、重复加载、导出 UUID 冲突、
准确性展示门控、跨策略同名事件隔离、无效数值 N/A、实际页面构建启动、分析任务失败状态和帧进度。
这些自动化测试不依赖下载模型或 GPU。

## 真实 HTTP 操作

控制台运行时实际执行：

```powershell
uv run python scripts/validate_console.py --run-analysis
uv run python scripts/validate_console.py --output outputs/console_validation/acceptance-final.json
```

首次完整验证 **7 组通过**，报告 `outputs/console_validation/report.json`；最终修改后复验
已有结果路径 **6 组通过**，报告 `outputs/console_validation/acceptance-final.json`。
脚本通过本机 Gradio HTTP API 执行了真实回调，检查：

1. 实验页面 TP/FP/FN、事件召回和机会召回逐项等于原始 `metrics.json`。
2. ABC 时间线读取正常；切换 `test_missed` 后保留三策略各一个遗漏机会。
3. 感知策略预览明确不使用真值，不显示准确性指标。
4. 合成 manifest 实际显示 5 条登记，调用既有校验器成功。
5. 时间查询 305ms 定位 frame 3 / 300ms，生命周期定位与保存的激活记录对应，创建独立 ZIP。
   `../../AGENTS.md` 被 Gradio 下拉框预处理拒绝；服务层还独立测试路径越界，未放宽输入限制。
6. 上传本机 `examples/tracking_fixture.mp4`，实际调用 CPU YOLO26n/BoT-SORT 与 Phase 2 风险重放。
   新结果：`outputs/console/analysis-20261009T152444-90575ea90455/`，10 帧、40 个检测、4 个追踪 ID、
   4 个风险事件、3 次首次提醒；上传原文件哈希未变。
7. 所有原 `outputs/phase3_final/` 文件在 HTTP 操作前后 SHA-256 完全一致。

上传夹具来自重复 `bus.jpg` 图片，是工程集成夹具，不能代表动态真实街景或独立道路评价。
模型权重、夹具 MP4、生成报告/媒体均按现有 `.gitignore` 管理，不当作随仓库提交的实验资源。

## 浏览器检查

通过 Codex 浏览器工具实际打开本机服务，观察四页页面和渲染图表；分别检查
1920×1080 与 1366×768 尺寸。发现并修复笔记本窗口的容器横向溢出；修复后页面文档宽度
未超过视口。宽表在自己的滚动区域查看，并非把整页扩宽。

- 视觉页：标注视频浏览器 `readyState=4`、时长 1 秒、480×640，点击播放后 `paused=false`。
  键盘输入 305ms 后实际显示 frame 3 / 300ms、4 个观察、1 条提醒及对应关键帧。
  原先只监听鼠标 release 的问题已修正为 change 回调，待处理请求保留最后一次。
- 对比页：真实展示 Synthetic 标签、4 个视频/13 秒、A/B/C 原值指标；取消 A 后完成筛选。
  时间线可见首次和升级提醒窗口、有效命中及遗漏；切换完全漏检视频后三条遗漏仍在。
- 分析页：原值指标表与统计图正常渲染，分层表和错误过滤接口通过 HTTP 验证。
- 数据集页：5 条 synthetic 登记可读，点击校验实际返回 5 视频、5 来源组、0 real、5 synthetic、
  总登记时长 15000ms 校验通过。

浏览器截图保存在 `outputs/console_validation/comparison-final.png`（最终版本）与
`outputs/console_validation/comparison-1920.png`（首次视觉检查）。
检查过程中修复的是布局和交互问题，不改动风险/匹配/指标算法。

## 限制与后续人工检查

没有足够的独立真实动态街景评价数据，本阶段不新增道路有效性结论。
播放器与分析时间控件独立，不能声称双向同步；JSONL 时间戳是研究时间依据。
当前验证为本机 Chromium 浏览器及短视频；Firefox/Safari、触屏、完整键盘遍历和长视频压力
仍需人工验证。500 MiB 上传限制不能替代 RAM/磁盘配额；媒体转码、缓存体积与数据量增长需要后续评估。
算法/权重/配置/锁文件发生变化后，新的 benchmark 仍需按 Phase 3 开发集和冻结协议流程建立版本。
停止于 Phase 4，未开发移动端、音频导航或新模型。
