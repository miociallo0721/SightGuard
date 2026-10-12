# 工具脚本

从仓库根目录运行脚本。用户入口优先使用 `uv run sightguard --help`；这里是数据准备、研究和维护工具。

| 用途 | 入口 |
| --- | --- |
| 文档维护 | `check_docs.py`，标准库即可，无需模型 |
| 可重建样例 | `generate_demo.py`、`generate_phase3_fixtures.py` |
| 公开数据准备 | `prepare_real_world.py`、`prepare_metric_dataset.py` |
| 深度 / 网格 / 走廊 | `run_depth.py`、`run_spatial_grid.py`、`run_corridor.py` |
| 运动研究 | `run_motion_quality.py`、`analyze_motion_quality.py` |
| 米制研究 | `fetch_metric_research.py`、`run_metric_research.py` |
| 音频依赖 | `prepare_openal.py` |

模型、输入前提和其余工具见 [研究文档](../docs/research/README.md)。Android 导出与设备工具在 [android/tools](../android/tools/)。`generate_phase3_fixtures.py` 用于声明的合成样例，不应用它重写人工标注或冻结基准。
