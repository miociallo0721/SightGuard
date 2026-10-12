# 第三方与资产说明

本文件索引仓库中已有的来源与许可记录，不为第三方资产新增授权，也不替代其原始许可。

| 组件或资产 | 仓库中的来源记录 |
| --- | --- |
| YOLO26n / YOLOE-26n | [模型清单](android/app/src/main/assets/yoloe_model_manifest.json)与 [YOLOE 技术说明](android/docs/yoloe_camerax_v07.md)，记录 Ultralytics AGPL-3.0 / Enterprise 条件 |
| SegFormer MiT-B0 | [来源说明](android/app/src/main/assets/licenses/README.txt)、[NVIDIA 原始许可](android/app/src/main/assets/licenses/SegFormer-NVIDIA.txt)、[MMSegmentation 许可](android/app/src/main/assets/licenses/MMSegmentation-Apache-2.0.txt) |
| Depth Anything V2 Small | [固定版本与许可字段](android/app/src/main/assets/depth_manifest.json) |
| 米制深度研究 | [模型清单](android/app/src/main/assets/metric_research/metric_manifest.json)与[许可文件](android/app/src/main/assets/metric_research/LICENSE.txt) |
| ADE20K 评测输入 | [来源与再分发边界](android/docs/semantic_furniture_v06.md)，原始照片和标注不随 Git/APK 发布 |
| Python 依赖 | [项目依赖](pyproject.toml)与 [uv.lock](uv.lock) |
| Android 依赖 | [Gradle 声明](android/app/build.gradle.kts) |

SegFormer 上游许可包含非商业研究/评估限制；框架许可证不自动替代权重或原始作品的条件。本仓库尚未提供自有源码的统一许可证，本次整理不擅自指定 MIT、Apache 或其他授权。对外发行前需由维护者明确源码和资产的分发条件。
