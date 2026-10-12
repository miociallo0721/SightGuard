# 公开数据评测与产品整理 — 2026-10-12

本轮在 `codex/dataset-product-readiness`、基线 `b3d0aeb` 上开发。先提交早些时候的 [PR #7](https://github.com/miociallo0721/SightGuard/pull/7)，再运行独立研究。执行窗口从 10:20 北京时间开始，最终硬截止 12:10；没有实体 Android 设备，不启动 AVD，也不将 PC 结果算作手机验收。

**结论：Metric3Dv2 Small 值得继续做手机测距研究，但不足以启用距离播报；地面语义与候选空间仍不能保证人体可通过。** 正式风险 / 环境声音、Android UI 和默认开关没有改变。没有新增路径规划、ARCore、GPS 速度融合或自动行进建议。

## 实际完成

- 官方 Metric3Dv2 ViT-Small 真实 PC 推理，保留原始宽高比、相机焦距缩放和光轴 Z 输出契约。
- 五个既有 TUM 序列各冻结 20 帧，共 100 帧，与同图像的历史 Hypersim 280 / 518 输出配对；输入和深度真值哈希重新核验。
- 从官方 NYUv2 labeled HDF5 按精确 HTTP Range 提取 32 个不同场景：8 个预留校准场景、24 个评估场景。先冻结名单再推理，没有根据输出选择成功图片或调整阈值。
- NYU 每张图实际运行 SegFormer、YOLO、原相对深度、Metric3D、Hypersim 280 / 518。真值不传给模型；原始有效深度与填补深度分别评价。
- 重用现有 WallMetricEstimator 与 LocalCorridorCandidateEvaluator；不改写评分、事件、光流或严格 D 门禁。
- 重写首页、Android / PC 使用说明和架构入口，增加状态、问题、版本 / 回退、贡献、隐私及文档索引。历史报告、冻结数据与旧包原样保留。

本轮新增 132 次 Metric3D 主推理，另有两次缺失 mask token 不变性对照；NYU 其余五个模型各运行 32 次。TUM 的旧模型输出是有来源的缓存对照，没有冒称本轮重新推理。

## 模型、来源与契约

| 模型 | 固定版本 / SHA256 | 本次用途 |
| --- | --- | --- |
| [Metric3Dv2 官方实现](https://github.com/YvanYin/Metric3D) | source `eb5b6fac0dc155e4e52f576e304fbf11655ff339`；[权重 revision](https://huggingface.co/JUGGHM/Metric3D/tree/80d2d1410afb4b23cd9d18c6be9144483d4b70b6)；SHA `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6` | `metric_depth_vit_small_800k.pth`，150,120,967 bytes，官方 ViT 616×1064 FP32 |
| [Depth Anything V2 Metric Small](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth/README.md) | source `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；Hypersim SHA `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545` | 280 / 518 官方等比例预处理，max_depth 20m |
| 既有 SegFormer / YOLO / 相对 Depth | 本机实际 ONNX SHA 保存在模型日志与交付 manifest | 与 Android 研究资产同文件；PC CPU ORT，不代表 Android 性能 |

Metric3D 遵循[固定源码的 hub 示例](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/hubconf.py)：RGB 0–255 归一化、等比例 resize / pad、去 padding、恢复来源尺寸、乘缩放焦距 / 1000。输出按预测光轴 Z 米解释，未经相机内参转换不称欧氏距离。没有以 GT 拟合比例或偏移。

官方 checkpoint 缺少且只缺少 `depth_model.encoder.mask_token`。最初 strict 加载失败记录保留；最终只允许这个已审计的缺项，拒绝其他缺失 / 多余键。每批首张真实图片将该 token 改为 1e6 后重复推理，深度逐值相同；推理路径未提供 masking 参数。没有任意忽略加载错误。

Metric3D 上游代码 BSD-2-Clause；独立权重再分发许可本轮未确认，不随 APK 或 Git 分发。Hypersim Small 来源许可见此前研究记录。NYU 官方页面提供公开研究下载和论文引用，本轮未找到明确独立再分发许可，因此只存本地、不给源码仓库附原图 / GT。不能将代码许可自动套用于权重或数据。

## 数据与真值

[NYU Depth V2 官方说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)中的 `rawDepths` 是带无效像素的米制深度，`depths` 是填补深度。运行时读取 names，确认 **NYU wall=21、floor=11**；这与 ADE20K wall=0、floor=3 不同。RGB / labels / depth 从 HDF5 还原为同一 480×640 坐标，不重复除以 1000 或 5000。

NYU 原文件 2,972,037,809 bytes，ETag `"b125b2b1-5aa5b95787a80"`。逐块校验 206、Content-Range、ETag 和本地 SHA256；没有完整下载，不能声称核对了发布方整文件 checksum。全文件下载过慢中止，改为有预算的分块；一次 TLS EOF 失败后保留日志，按原名单恢复，最终 32/32 完成。

NYU 使用官方 toolbox `camera_params.m` 的 RGB fx=518.8579011745019、fy=519.4696111212749、cx=325.5824494111903、cy=253.7361663340047。TUM 使用对应 fr1 / fr3 官方内参。没有另做畸变校正，属于限制；焦距来自公开标定，不来自深度真值。

[TUM RGB-D 格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)的 registered depth 是 uint16 / 5000，0 无效；CC BY 4.0。沿用既有五个序列和原 RGB / 深度关联：fr1_xyz、fr1_rpy、fr3_nostructure_notexture_far、fr3_structure_texture_far、fr3_nostructure_texture_near_withloop。100 帧名单 SHA256 `1d53cf8feb2f43ff74cc4539c92af7b5938f6b00df7075d596761efb680631ac`；由新冻结脚本重新生成后，内容与推理前的名单完全一致。

NYU 每个 still 是独立会话，不能重复一张图制造三帧确认。校准场景没有用于拟合或调参；这一分组也不能证明模型训练时从未见过 NYU。因此结果是有限子集诊断，**不是官方 benchmark 成绩、零样本泛化或手机精度认证**。

## 米制误差

以下为像素加权误差，没有尺度对齐或按结果删除样本。MAE / RMSE 单位米，AbsRel 无单位。

| 数据 / 区域 | 模型 | MAE | RMSE | AbsRel |
| --- | --- | ---: | ---: | ---: |
| NYU 24 评估场景，原始有效 GT，全图 | Metric3D | 0.2830 | 0.4812 | 0.1021 |
| 同上 | Hypersim 280 | 1.4155 | 1.6753 | 0.5597 |
| 同上 | Hypersim 518 | 0.6354 | 0.9263 | 0.2491 |
| NYU 独立 GT 墙面，原始有效深度 | Metric3D | 0.3036 | 0.4899 | 0.0879 |
| 同上 | Hypersim 280 | 1.3965 | 1.5830 | 0.4645 |
| 同上 | Hypersim 518 | 0.6506 | 0.8900 | 0.2142 |
| TUM 100 帧，全图 | Metric3D | 0.3926 | 0.5217 | 0.2280 |
| 同上 | Hypersim 280 | 0.4975 | 0.6891 | 0.3491 |
| 同上 | Hypersim 518 | 0.5051 | 0.6376 | 0.3050 |
| TUM 预测 wall 区域，非语义真值 | Metric3D | 0.5216 | 0.6283 | 0.2929 |
| 同上 | Hypersim 280 | 0.4524 | 0.7630 | 0.2072 |
| 同上 | Hypersim 518 | 0.5852 | 0.7453 | 0.2833 |

NYU 原始有效深度全图有 4,741,252 个可比较像素，GT wall 有 1,118,451 个；三模型在这些像素的有效输出比例均为 1。有效输出不代表预测正确，也不代表整个图像都有传感器真值。

填补深度另算：NYU Metric3D 全图 MAE 0.4428、RMSE 1.1600，GT wall MAE 0.3299、RMSE 0.5824；不能用填补消除反光 / 玻璃等传感器空洞后声称实测精度。交付 CSV 保留每图误差分位数，JSON 保留场景和 GT 距离分组，包含空组 / 无有效像素，而非只给全局平均。

Metric3D 在该 NYU 子集明显改善，但 TUM 预测墙面 MAE 仍差于 Hypersim 280，不能宣布全面胜出。缺少手机卷尺测距、胸前视角、独立部署域校准和足够边界负例，**不生成有准确率保证的距离区间，不启用米制语音**。WallMetric 的 P10 / P90 仍只是墙面像素空间分布。

## 地面语义、墙面统计与候选

NYU 24 评估场景在固定 0.55 置信阈值下，wall 像素 precision / recall / IoU 为 **0.7679 / 0.8156 / 0.6543**；floor 为 **0.8162 / 0.8609 / 0.7211**。0.35 对照及 8 个校准场景单独保留，没有根据这些数值修改生产阈值。

失败 `nyu_0457 / reception_room_0002` 的 floor precision 仅 **0.1642**（9,447 TP、48,070 FP）；`nyu_0278 / study_room_0006` 为 **0.5090**。图像复核可见玻璃 / 远处区域和拥挤家具场景中的地面预测覆盖问题。标签边界和未标注像素也需考虑；这些是像素级诊断，不是系统事件误报率或通行准确率。

32 张 NYU 图像运行原候选评估器：30 INSUFFICIENT_EVIDENCE、2 TIED，严格 D **32 ABSTAIN**。每个原始左 / 中 / 右候选附带独立 GT floor / wall / unlabeled 覆盖，GT 只在候选返回后追加。没有 SUPPORTED_CANDIDATE、SAFE、PASSABLE、真实宽度或路径指令。不能以单图不足连续确认的结果估算动态候选召回率。

Metric3D 接入既有墙面统计器的离线处理成本画像：132 条中 6 条 PREDICTED_METRIC_DEPTH、126 条 ABSTAIN；88 条同时受到源龄门禁限制。6 条通过记录的模拟年龄为 494.1–499.9ms，几乎没有排队余量。该年龄是分别运行的语义 + 深度实测处理成本相加，**是模拟串行时序，不是 CameraX capture-to-consume 实测**。保留 500ms 门禁；没有为了让更多记录通过而放宽期限。

墙面接近算法本轮没有新充分正负例验收，沿用既往 700 帧全 ABSTAIN 的限制。新米制数值不能直接充当接近趋势。纯旋转、身体朝向、低纹理、手机局部运动与实际前行仍需独立实验，自动接近提示关闭。

## PC 性能与工程检查

硬件 i7-12650H / RTX2050 4GB / RAM 16GB。每个模型串行，CPU / ORT 两线程，GPU fraction 0.65，研究进程降优先级；采样点至少保留 1536MB RAM / 5GB 磁盘。不启动模拟器，不终止用户其他进程。

| NYU 32 图，排除首张冷启动 | 执行后端 | 推理 p50 / p95 |
| --- | --- | ---: |
| Metric3D 616×1064 FP32 | CUDA | 422.42 / 427.58ms |
| Hypersim 280 | CUDA | 27.65 / 27.79ms |
| Hypersim 518 | CUDA | 101.66 / 102.16ms |
| SegFormer 256 | CPU ORT | 34.83 / 36.24ms |
| YOLO 320 | CPU ORT | 11.31 / 12.33ms |
| 原相对 Depth 280 | CPU ORT | 147.89 / 152.78ms |

这是各自真实运行配置，CPU / CUDA 不混称等后端性能排名。Metric3D TUM p50 / p95 为 421.52 / 428.81ms；记录到 GPU allocated 峰值约 819MB、reserved 1264MB，最小采样可用 RAM 2151MB。没有全系统连续峰值监控、手机 FPS / PSS、耗电或热回归；不能推断 nova12 / Redmi 表现。

- 最终默认 pytest：**643 passed、27 skipped**；新增 29 项覆盖来源 / 缓存 / 预算 / 时限、几何、指标与冻结名单。跳过均为原有可选模型 / 环境测试，不删除或弱化有效测试。
- Ruff check、全仓格式检查和 diff whitespace 检查通过。五个旧工具仅格式调整，前后 AST hash 相同。
- README 的合成 assess 命令实际重放 5 帧、生成 1 条结构化提醒；不播放声音，不算真实识别验收。CLI help 和文档本地链接另行核对。
- 与官方示例的 RGB 归一化逐值一致；合成深度场经 OpenCV / 官方 Torch 双线性恢复的最大差 0.000429m 以下。该检查验证几何数值，不是模型 / ONNX / Android 精度对照。
- 新增独立 `research/dataset-product/uv.lock`；本轮复用已安装 GPU 环境并隔离补充依赖，版本快照保存。未重新完整安装该锁定环境，不宣称干净克隆已验证。

## 复现入口

先按[此前研究说明](../../android/docs/metric_corridor_research.md#复现)准备原 ONNX、官方 Depth Anything 源码 / 权重及 TUM 两份冻结 manifest 和原模型缓存。本轮不会为重跑改写旧交付。

以下在仓库根 PowerShell 执行，路径可替换为本机资产目录。`$stopUtc` 必须是本次复现明确允许的未来截止时间；所有输出使用新目录，脚本拒绝覆盖推理结果。新环境准备需要联网和 CUDA 驱动，不能当作默认离线测试。

```powershell
uv sync --locked --project research/dataset-product
$py=(Resolve-Path research/dataset-product/.venv/Scripts/python.exe).Path
$env:PYTHONPATH=(Join-Path (Get-Location) src)
$oldCache='D:/program/SightGuard/models/metric-corridor'
$cache='D:/program/SightGuard/models/dataset-product-sprint-20261012'
$delivery='android/deliveries/your-new-comparison'
$stopUtc=(Get-Date).ToUniversalTime().AddHours(1).ToString('o')
New-Item -ItemType Directory -Path $delivery
& $py scripts/fetch_dataset_research_model.py --cache $cache --stop-utc $stopUtc
& $py scripts/prepare_nyu_research.py --output "$cache/nyu-new" --cache "$cache/nyu-ranges" --count 32 --stop-utc $stopUtc
# 网络暂时失败时，保留原冻结名单，用相同参数加 --resume；不要另选成功样本
& $py scripts/finalize_nyu_manifest.py --extracted "$cache/nyu-new" --output "$delivery/nyu32-manifest.json"
& $py scripts/freeze_tum_comparison.py --manifest "$oldCache/sample-manifest.json" --manifest "$oldCache/extension-manifest.json" --output "$delivery/tum100-manifest.json"
& $py scripts/run_dataset_depth_comparison.py --manifest "$delivery/tum100-manifest.json" --model-cache $cache --existing-cache $oldCache --output "$delivery/tum100" --phase metric3d --stop-utc $stopUtc
foreach ($phase in @('metric3d','semantics','relative','hypersim280','hypersim518')) {
  & $py scripts/run_dataset_depth_comparison.py --manifest "$delivery/nyu32-manifest.json" --model-cache $cache --existing-cache $oldCache --output "$delivery/nyu" --phase $phase --stop-utc $stopUtc
  if ($LASTEXITCODE -ne 0) { throw "Phase failed: $phase" }
}
& $py scripts/evaluate_dataset_depth.py --delivery $delivery --previous-delivery android/deliveries/metric-corridor-research --nyu-manifest "$delivery/nyu32-manifest.json" --output "$delivery/evaluation-final" --stop-utc $stopUtc
& $py scripts/evaluate_wall_metric_outputs.py --delivery $delivery --previous-delivery android/deliveries/metric-corridor-research --output "$delivery/wall-metric-results.jsonl" --stop-utc $stopUtc
```

当前本机实际运行资料在 `android/deliveries/dataset-product-sprint-20261012/`，含冻结输入名单、每图 CSV / JSONL、来源哈希、原始失败日志、性能 / 版本快照、失败图、验证汇总、最终报告与回退说明。模型、原数据和生成输出由 Git ignore 排除，不作为 PR 附件。

本轮没有改 Android 应用源码或模型资产，没有新 Android 包、设备推理或声音 onStart 验收。保留[原稳定 / 实验 APK 及哈希](releases.md)，不为文档更新重新命名旧 APK 冒充新增手机能力。下一步优先做目标手机已量测墙距、端侧时延，以及有独立地面 / 净空标注的负例；仅增加更大的深度模型不能解决安全通行问题。
