# 桌面工具

桌面端用于本地视频分析、摄像头体验、结果重放和研究。Python 包版本与 Android 版本分别管理。

## 环境与模型

需要 Python 3.12 和 uv。从仓库根目录执行：

```powershell
uv python install 3.12
uv sync --locked
uv run sightguard --help
```

首次联网准备官方 YOLO 权重，后续可离线运行：

```powershell
uv run python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt')"
```

使用前阅读[模型来源与使用条件](../development/models.md)。默认分析不会自动下载模型；模型放在本机 `models/`，不提交 Git。

## 产品网页

```powershell
uv run sightguard ui --port 7860 --open-browser
```

打开 `http://127.0.0.1:7860`，选择本地 MP4 并开始分析。服务默认只监听本机，不要直接暴露到公网。网页的方向与马达展示是演示内容，不是 Android 通行指令。

## 命令行分析

```powershell
uv run sightguard analyze --input <本地视频.mp4> --output-dir outputs/my-video --config configs/default.yaml
uv run sightguard replay --input outputs/my-video/tracks.jsonl
uv run sightguard assess --tracks outputs/my-video/tracks.jsonl --output-dir outputs/my-assessment
```

`analyze` 生成感知记录和标注视频；`assess` 读取已有记录，不重新运行模型。每次使用新的输出目录，以保留已有结果。默认 PC 配置筛选 person、bicycle、motorcycle；它不等于 Android 的类别配置。

实时摄像头：`uv run sightguard live`。研究控制台按需启动：`uv run sightguard ui --research --port 7861`。先启动需要的单个服务，避免同时加载多个模型。

详细参数见 `--help`；更多内容见[实时摄像头](live.md)、[标注工作台](annotation_studio.md)和[数据契约](../reference/pc-data-contracts.md)。
