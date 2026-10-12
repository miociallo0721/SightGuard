# 模型准备与来源

模型权重和导出二进制不提交 Git，也不在应用运行时自动下载。交付包必须使用与代码清单一致的资产，不能把同名的另一份 ONNX 文件当作兼容模型。

## Android 资产

以下文件放在 `android/app/src/main/assets/`：

| 文件 | 用途 | 契约位置 |
| --- | --- | --- |
| `yoloe26n.onnx` | 默认目标模型，固定 90 类提示 | `yoloe_model_manifest.json` |
| `yolo26n.onnx` | 设置中的旧目标模型 | `model_manifest.json` |
| `segformer_mit_b0_ade20k_256.onnx` | 基础环境语义及降级模型 | `SpatialDetector.MODEL_SHA`、`ade20k_labels.json` |
| `segformer_mit_b0_ade20k_384.onnx` | 更细的环境语义，受负载控制 | `SegFormerModels.DETAIL_SHA` |
| `depth_anything_v2_small_280.onnx` | 相对深度辅助/研究 | `depth_manifest.json` |

320 分辨率和 `metric_research/` 下的米制模型是独立研究资产。米制研究默认关闭，不是墙面存在提示的前置条件。缺少 384 模型会限制新增家具自动提醒的资格，不能把降级包标成能力完全相同的包。

## 重建步骤

导出依赖固定在 [requirements-model-export.txt](../../android/requirements-model-export.txt)，建议创建独立 Python 3.12 环境，避免改变桌面端锁文件：

```powershell
uv venv --python 3.12 android/.export-venv
uv pip install --python android/.export-venv/Scripts/python.exe -r android/requirements-model-export.txt
```

各导出工具需要先准备原权重和对照输入；它们不是从空仓库一键生成所有资产的安装器。

- YOLO26n：本地 `models/yolo26n.pt`；导出工具 [export_model.py](../../android/tools/export_model.py)。
- YOLOE：先生成冻结提示模型 `android/deliveries/yoloe26-quick/yoloe26n-fixed.pt`，再运行 [export_yoloe.py](../../android/tools/export_yoloe.py)。权重来源、公开输入准备和固定提示步骤见 [YOLOE 模型说明](../../android/docs/yoloe26_quick_test.md)。
- SegFormer：冻结 OpenMMLab 权重 `models/path_relevance/segformer_mit_b0_ade20k_160k.pth`，使用 [export_segformer.py](../../android/tools/export_segformer.py)。256 导出默认写入 `android/spatial-models/`，通过同输入核验后再放入应用 assets；384 导出见[家具语义说明](../../android/docs/semantic_furniture_v06.md)。
- 相对深度：本地固定模型配置与权重，使用 [export_depth.py](../../android/tools/export_depth.py)，来源见 [depth_manifest.json](../../android/app/src/main/assets/depth_manifest.json)。

导出环境变化可能导致文件 SHA 改变。必须先检查预处理、同输入数值结果、输出语义和设备推理，再有意更新契约；不要删除运行时校验。

## 来源与使用条件

模型来源、许可证和固定版本以清单及[第三方说明](../../THIRD_PARTY_NOTICES.md)为准。SegFormer 的原始许可保留研究/评估限制，不能因框架使用 Apache-2.0 就把整包视为可任意商业分发。公开数据图片不打入正式 APK。
