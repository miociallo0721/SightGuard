# 隔离研究环境

本目录保存独立研究任务的依赖声明和锁文件，不参与 Android 默认推理或桌面正式提醒。

- `metric-corridor/`：米制深度与局部空间研究的 Python 环境。
- `metric-corridor/android-ort/`：Android ONNX 导出相关隔离环境。

算法入口在 `scripts/`，研究说明见 [研究文档](../docs/research/README.md)与 [Android 米制研究](../android/docs/metric_corridor_research.md)。不要将研究依赖合并进默认环境，也不要把权重、数据集或生成报告放进 Git。
