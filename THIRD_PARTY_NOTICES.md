# 第三方与资产说明

本文件索引仓库中已有的来源与许可记录，不为第三方资产新增授权，也不替代其原始许可。

## 自有源码

Copyright © 2026 SightGuard contributors.

除另有说明的第三方内容外，本项目有权授权的原创源码按 [GNU Affero General Public License 第 3 版](LICENSE)发布，SPDX 标识为 `AGPL-3.0-only`。许可证原文来自 GNU 官方，不修改条款。源代码见 https://github.com/miociallo0721/SightGuard 。

这是自有源码的授权选择，不是将整个仓库内所有资产或 APK 中全部组件统一重新授权。第三方代码、模型权重、公开数据及其他另行许可的素材继续适用其原始条款。

## 第三方内容

| 组件或资产 | 仓库中的来源记录 |
| --- | --- |
| YOLO26n / YOLOE-26n | [模型清单](android/app/src/main/assets/yoloe_model_manifest.json)与 [YOLOE 技术说明](android/docs/yoloe_camerax_v07.md)，记录 Ultralytics AGPL-3.0 / Enterprise 条件 |
| SegFormer MiT-B0 | [来源说明](android/app/src/main/assets/licenses/README.txt)、[NVIDIA 原始许可](android/app/src/main/assets/licenses/SegFormer-NVIDIA.txt)、[MMSegmentation 许可](android/app/src/main/assets/licenses/MMSegmentation-Apache-2.0.txt) |
| Depth Anything V2 Small | [固定版本与许可字段](android/app/src/main/assets/depth_manifest.json) |
| 米制深度研究 | [模型清单](android/app/src/main/assets/metric_research/metric_manifest.json)与[许可文件](android/app/src/main/assets/metric_research/LICENSE.txt) |
| ADE20K 评测输入 | [来源与再分发边界](android/docs/semantic_furniture_v06.md)，原始照片和标注不随 Git/APK 发布 |
| Python 依赖 | [项目依赖](pyproject.toml)与 [uv.lock](uv.lock) |
| Android 依赖 | [Gradle 声明](android/app/build.gradle.kts) |

SegFormer 上游许可包含非商业研究/评估限制；框架许可证不自动替代权重或原始作品的条件。该限制与 AGPL 允许的用途不同，不能通过增加根目录 LICENSE 或分列许可说明来认定组合分发已合规。对外发行前仍需核对代码与模型的实际来源、衍生关系及兼容性；必要时替换受限组件或取得额外授权。
