# 示例数据

不提交视频。生成本地合成视频：`uv run python scripts/generate_demo.py`。
该视频只检验读写链路，不代表真实风险或检测质量。

生成包含真实目标的跟踪冒烟短片：`uv run python scripts/generate_tracking_fixture.py`。
它使用本地已安装 Ultralytics 随包的 `assets/bus.jpg`，缩放为 480×640 后重复 10 帧，
10 FPS、mp4v。生成 `tracking_fixture.mp4` 及 `.provenance.json`（含原图与视频哈希）。
它可以检查目标 ID 与帧间关联接口，但静态重复图像不能评估真实运动下的跟踪准确率。
