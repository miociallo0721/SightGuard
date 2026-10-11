# Depth Integration v1

2026-10-11。独立离线相对深度证据。默认 YOLO/BoT-SORT、Risk Engine、EventManager、
AlertScheduler、AlertEvent、HRTF、SegFormer/Path Relevance、Laya Shadow 不变。
没有实现空间网格、Corridor Planner、新风险公式或导航反馈。

## 模型与许可

使用 [官方 HF Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/tree/5426e4f0f36572d16453bbda7a8389317b1bef99)，
`DepthAnythingForDepthEstimation` / DINOv2 ViT-S + DPT，实际参数 24,785,089，FP32。
revision：`5426e4f0f36572d16453bbda7a8389317b1bef99`。

- `model.safetensors`：99,173,660 bytes；SHA-256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。
- `config.json`：SHA-256 `c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75`。
- `preprocessor_config.json`：SHA-256 `d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84`。
- 固定 model card：SHA-256 `319566441daaac0c5ed83e75a8fd19bae0863f3275e23f147c06fe0906edf6fb`。

[官方源码仓库](https://github.com/DepthAnything/Depth-Anything-V2)的源码 LICENSE 为 Apache-2.0，
README 明确 Small checkpoint 为 Apache-2.0；固定 HF model card 同样标为 Apache-2.0。
Base/Large/Giant 的许可不适用于本次 Small。权重没有随 SightGuard 分发；未来分发仍须保留
相应许可、通知与来源。本次未复制上游实现，只调用已有 Transformers。
依赖分别沿用自身许可（Transformers Apache-2.0、PyTorch/NumPy BSD 系列、
OpenCV Apache-2.0、Pillow HPND）；这些许可不替代模型与 UrbanEgo 数据许可。

新增 `depth` optional extra 与既有 `path-relevance` 共用 Transformers，`uv.lock`
只增加 extra 映射，没有改变任何已有解析版本。CUDA 研究复用已有隔离环境；其实际
Torch 版本与主 `.venv` 不同，全部实测版本写在运行元数据，不宣称 GPU 环境由主 uv.lock 创建。

## 预处理、语义与对齐

输入是非空 uint8 H×W×3 原始 BGR，明确复制转换为 RGB，不修改调用者的图像。
`DPTImageProcessor` 使用本地固定配置：PIL bicubic、保持长宽比、尺寸取 14 的倍数、
`/255` rescale、ImageNet mean/std；不填充。`input_size=280` 在本批次 1280×720 图像上
得到 **504×280** 输入。配置表示处理器目标尺寸，不是强制正方形或输出原图尺寸。
[HF 文档](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2)说明底层模型与
`predicted_depth` 接口；官方仓库也提示 HF 与 OpenCV 预处理/恢复可能有数值差异。
不宣称本适配器与原生 OpenCV 实现逐像素相等。

直接读取 `outputs.predicted_depth`，不用 pipeline 渲染后的 PIL `depth` 图片作为数据。
原生预测是未归一化相对逆深度式数值；**大值代表相对更近，小值代表相对更远**。
这个方向除配置确认是 relative 以外，还在真实 V05 第 0 帧、原始 RGB 选取的近墙和远车
区域做了数值检查，详见验证报告；不是根据热力图颜色推断。

原生数组与 `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)`
恢复到原图 H×W 的数组都保留。恢复可能在边界产生负值，原样保存，不裁剪、不求倒数。
二维数组方向均为 `[y, x]`，source 像素坐标来自原图；source-aligned 数组可与原图 bbox
在同一坐标域读取。双三次恢复只解决空间尺寸，不能恢复输入下采样丢失的细节。

数值无米单位、无已校准概率，跨帧/视频尺度不可假定可比。禁止根据预测相减计算真实速度、
TTC 或固定米制危险阈值。每帧 min/max 的显示缩放仅用于 JPEG，不能反馈到算法数据。

## 接口与生命周期

`perception/depth/schema.py` 只依赖标准库。模型模块独立加载 optional runtime；
没有 import 时加载、下载、音频或决策调用。

```python
from pathlib import Path
from sightguard.perception.depth.model import DepthEstimator
from sightguard.perception.depth.schema import DepthModelConfig

estimator = DepthEstimator(
    DepthModelConfig(
        checkpoint_dir=Path("models/depth_anything_v2_small"),
        device="cpu",
    )
)
estimator.initialize()  # local files only; verify all three artifact hashes
try:
    prediction = estimator.infer(
        frame_bgr,
        video_id="urbanego_run8_05",
        stream_id="urbanego_run8_05",
        frame_index=0,
        timestamp_ms=0.0,
        timestamp_source="video_pts",
        source_width=1280,
        source_height=720,
    )
    result = prediction.result  # serializable metadata
    depth = prediction.depth_map  # float32 or None; require result.status == "valid"
finally:
    estimator.close()
```

`initialize()` 幂等，不逐帧加载。显式 CUDA 不可用时抛 `DepthModelUnavailableError`；
auto 按实际环境选 CUDA/CPU。仅 CUDA OOM 可按配置降到 CPU，一次 CPU retry；不无限重试。
失败帧返回当前帧 UNKNOWN，不保留上一帧预测。`close()` 释放模型、处理器并清理空闲 allocator
缓存，不干预其他模型共用的 cuBLAS 工作区。结果数组在 CPU，卸载后仍可保存。

## UNKNOWN 与有效性

`status` 为 valid/unknown；`inference_status` 独立区分 model_unavailable、image_read_failed、
inference_failed、completed。工程原因包括 invalid_input、invalid_output_size、non_finite_output、
insufficient_output_variation、CUDA OOM/fallback 失败等。视频解码、hash 或关联失败还会写入
该视频 summary 的 failed/error，其他视频继续处理，不假造未解码帧成功。

NaN/Inf、无效尺寸不保存可消费数组。有限但过于平坦的输出可以保存诊断矩阵，仍然 UNKNOWN。
变化检查使用 `(p99-p01)/max(abs(output), 1e-12)` 与配置阈值，检查原生及恢复结果。
它只检测明显工程异常，不能识别所有低纹理、反光、遮挡、细物体失败；valid 不表示模型可靠。
没有像素置信度、预测概率或安全标记。UNKNOWN 永远不能解释为 free/safe。

## 文件契约与读取

每个新 batch 目录必须不存在，拒绝覆盖甚至已有空目录。每视频独立目录：

- `depth_frames.jsonl`：schema `sightguard.depth.v1`，既有 video_id/stream_id、frame_index、
  timestamp_ms/source、source 宽高、模型/revision/hash/device、input/native/aligned shape、
  原始数值统计、timings、status/reasons/error、NPZ/JPEG reference、NPZ SHA-256。
- `depth/000000.npz`：`depth` 为 source-aligned float32，`model_output` 为原生 float32；
  `np.savez_compressed` 无损压缩，无 float16 转换或显示归一化。
- `previews/000000.jpg`：原始 RGB 内容与相对深度显示，时间、状态、尺寸及统计；仅诊断。
- `alignment.jsonl`：关联 tracks/risk/scene 的行索引、帧键及已存在 event IDs；不进行风险融合。
- `summary.json` / `anomalies.json`：每视频计数、失败与 UNKNOWN 索引、来源哈希和性能。
- batch `run_metadata.json`：manifest/hash、配置、模型来源、软件/硬件、源码哈希、热身、
  时间范围、全部视频、CPU fallback 状态、显存及卸载后资源、失败信息。

消费示例：读取 JSONL 的某一行，以三键确认目标帧，再调用
`load_depth_map(video_directory, row)`；此函数拒绝 UNKNOWN、错误 schema/语义、路径逃逸、
hash 不匹配及损坏尺寸/dtype/非有限数值。`DepthFrameResult.matches_frame()` 要求
video_id + frame_index + timestamp_ms 精确一致。引用相对于视频目录，返回 H×W float32。
也可用 `np.load(path, allow_pickle=False)["model_output"]` 检查原始预测。

## 时间和数据来源

读取 real-world v0.1 manifest，核对文件 SHA-256、容器 FPS/尺寸、tracks/risk 帧数与单调时间。
采样沿用 v04 的 `round(fps*interval_seconds)` 帧步长，额外拒绝非整数帧间隔。
本批次每 10 帧一次，0–290，共 30 帧；时间 **读取 tracks**，不按推理完成时间生成。
同时核对 scene_frames 的采样索引/时间与当前解码 PTS。tracks 使用 fallback 时保留该来源。
没有跨帧插值，没有 TTL 偷换，没有把旧深度当当前帧。
MP4 时间与原始传感器 capture time 的映射仍 unavailable，原 manifest 说明原样保留。

## 复现命令

CPU 环境先按 uv.lock 安装 optional extra；若本机正在运行 sightguard.exe，先由使用者正常退出
再同步，或使用已具备依赖的隔离解释器。所有命令在仓库根目录执行：

```powershell
uv sync --extra depth
uv run --extra depth python scripts/prepare_depth.py
uv run --extra depth python scripts/run_depth.py --device cpu --output-dir outputs/depth_integration/cpu_new
```

本机已验证 CUDA 环境与共享工作区库的路径顺序：

```powershell
& models/hybrid/env-structured-gpu/Scripts/python.exe scripts/run_depth.py --workspace-dependencies --device cuda:0 --output-dir outputs/depth_integration/new_run
uv run --no-sync python scripts/validate_depth.py --run-dir outputs/depth_integration/new_run --output-dir outputs/depth_validation/new_validation
```

`--video-id urbanego_run8_05` 可只处理一段；`--config/--manifest/--baseline-dir/--scene-dir`
可选择已有输入。不会自动下载。准备脚本仅下载固定 revision 的四个文件，验证 hash/大小，
不覆盖已经存在的损坏文件，不克隆 LFS 仓库。权重和输出都由 `.gitignore` 排除。

## 下一阶段接口与限制

Spatial Grid Fusion 可以复用 source 像素坐标、原始逆深度数组、video/stream/frame/time 键、
每源独立有效性、模型 provenance 和关联索引。本轮只提供证据：未来需独立设计网格聚合、
遮挡/相机运动/跨帧尺度处理与未知格规则，再通过独立标注验证。低频插值仅可作为未来明确
带来源/时效的估计方案。没有实现融合、可通行分类、提醒调整、米制测距或碰撞预测。
新 src 文件改变既有全源码 benchmark fingerprint；旧冻结协议与历史结果保留，未自动重封。
