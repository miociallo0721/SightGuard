# PC 使用与离线重放

PC 工具用于演示、复现和评测。其模型参数、BoT-SORT 跟踪器和可选 OpenAL 音频与 Android 实现不同；不能用 PC 性能推断手机性能。

## 安装

从仓库根目录执行，需要 Python 3.12 与 uv：

```powershell
uv sync --locked
uv run sightguard --help
```

依赖由 `uv.lock` 固定。默认配置 CPU；首次依赖和权重准备需要联网，准备好后分析使用本地文件。

## 无模型事件重放

```powershell
uv run sightguard assess --tracks tests/fixtures/phase2/static_occlusion.jsonl --config configs/risk.yaml --stream-id synthetic-01 --output-dir outputs/first-run/risk
```

输入是合成检测记录，只验证规则、事件和日志链路。`assess` 不重新加载检测模型、不播放音频。输出包括 `risk_frames.jsonl`、`events.jsonl`、`alerts.jsonl` 和运行元数据。输出目录已有同名产物时拒绝覆盖，重新运行请使用新目录。

## 准备 YOLO 与分析本地视频

权重未包含在 Git。以下是显式的联网模型准备命令，不属于离线测试：

```powershell
uv run python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt')"
uv run sightguard analyze --input examples/your-video.mp4 --output-dir outputs/video-01 --config configs/default.yaml
uv run sightguard replay --input outputs/video-01/tracks.jsonl
uv run sightguard assess --tracks outputs/video-01/tracks.jsonl --config configs/risk.yaml --output-dir outputs/video-01/risk
```

`examples/your-video.mp4` 是已合法取得的本地 MP4，并非仓库自带视频。下载后记录模型 SHA256；默认 analyze 不自动下载缺失权重。

`configs/default.yaml` 当前为 CPU、640 输入、置信度 0.25、COCO person / bicycle / motorcycle。按实际实验设置类别，不要误以为这等同 Android 的全类别 320 输入。`--model`、`--device` 等选项见命令帮助。

分析元数据记录输入 / 模型 / 配置哈希、时间来源、软件环境和失败状态。评估前检查运行是否完整结束；空检测不等于无风险。精确时间应以 JSONL 为准，标注视频是无原音频的可视化副本。

## 本地界面与摄像头

```powershell
uv run sightguard ui --port 7860 --open-browser
# 按需研究控制台，勿在资源紧张时与大型评测并行
uv run sightguard ui --research --port 7861
# 摄像头编号依本机实际设备确定
uv run sightguard live --camera 0 --device cpu --duration 30
```

默认网页为 FastAPI 演示页；`--research` 为 Gradio 控制台。网页方向演示和浏览器语音试听不是正式导航功能。

实时规则提醒复用原事件核心。PC 实际音频需显式启用 `--spatial-audio` 并准备 OpenAL，详见[音频说明](spatial_audio_architecture.md)。生成 `AlertEvent` 不等于已播放。

## 测试与研究

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

默认测试不要求网络或 GPU；opt-in 集成测试需按专项文档准备本地权重，记录跳过原因。合成测试通过不能代表真实道路效果。独立标注与协议见[评测方法](phase3_methodology.md)，额外模型入口见[文档索引](README.md)。

长会话仍有 PC 已结束事件未释放的问题。在修复并完成回归前，不应将无人值守长时运行当作已通过的产品能力，见[问题 P0-04](product/known-issues.md#p0-04-pc-长会话事件保留未有界化)。
