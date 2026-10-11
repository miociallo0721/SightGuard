# Path Relevance v1：实验与验证记录

记录日期：2026-10-11。最终可复核批次为 `outputs/path_relevance/v04/`。输出目录受 `.gitignore` 忽略，不提交原始视频、权重、叠加视频帧或批次 JSONL。

## 运行环境和输入

- 模型：OpenMMLab MMSegmentation model zoo 的 SegFormer MIT-B0 / ADE20K / 512×512 / 160k。
- checkpoint SHA-256：`8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`；源链接见 `run_metadata.json`。文件大小 15,072,692 bytes。
- 当前设备：NVIDIA GeForce RTX 2050 Laptop GPU，4GB 显存；实际运行设备 `cuda:0`，本批次没有 CPU fallback。
- 运行环境：Python 3.12.15、PyTorch 2.10.0+cu130、CUDA runtime 13.0、Transformers 5.19.0、NumPy 2.5.3、OpenCV 4.14.0.94。运行元数据同时记录软件版本、设备和模型哈希。
- 研究配置：`configs/path_relevance.yaml`，输入宽度 512，采样间隔 0.5 秒，分割像素置信度至少 0.35，候选区域至少占 0.5% 面积、从下部 18% 区域延伸且纵向至少 35%；结果 TTL 500 ms；Path Relevance 的 MEDIUM/HIGH 门限分别 0.20/0.55。没有调高 Risk Engine 阈值。
- 输入：`datasets/real_world/manifests/dataset.json` 中八段 UrbanEgo 片段，共 120 秒，20 FPS、720p、2,400 帧，均为同一原始拍摄的 development 片段。复用 `outputs/real_world/v01/<video_id>/perception/tracks.jsonl` 和 `assessment/{risk_frames,events,alerts}.jsonl`，不重新运行 YOLO26n / BoT-SORT。
- manifest SHA-256：`b4de54f10c7236968532792a05ad1997b4033f0f7c9ed9d5144421185bcb1d42`。每个视频的原视频、tracks、risk frames、events、alerts、model 哈希都保存在批次 `run_metadata.json`。
- 标签状态：`datasets/real_world/annotations/visual_tasks.json` 为 `pending_human_annotation`；8 个视觉任务 `human_attributes` 均为 null。没有真实准确率计算。

## 全量结果

完整批次成功完成 8/8 视频，按 2 Hz 采样 240 帧，其中 238 帧产生有效候选语义地面区域，2 帧为 UNKNOWN。共输出 1,123 个目标观察；Path Relevance 分数有定义的 741 个，按观察加权的均值为 0.319。等级计数为 HIGH 125、MEDIUM 376、LOW 240、UNKNOWN 382。所有有效 Baseline `alerts.jsonl` 哈希均与运行前一致，Baseline AlertEvent 共 257 条（HIGH 120、MEDIUM 137）。

| 视频 | Baseline AlertEvent | 有效场景帧 / 30 | 目标观察 | PR 分数均值（已定义项） | HIGH / MEDIUM / LOW / UNKNOWN | 样本处理均值（ms） |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| V01 | 33 | 30 | 143 | 0.296 | 13 / 65 / 31 / 34 | 67.3 |
| V02 | 55 | 30 | 211 | 0.378 | 31 / 95 / 38 / 47 | 50.9 |
| V03 | 34 | 30 | 144 | 0.073 | 0 / 15 / 75 / 54 | 51.4 |
| V04 | 20 | 30 | 108 | 0.070 | 0 / 1 / 7 / 100 | 42.6 |
| V05 | 36 | 30 | 140 | 0.288 | 17 / 34 / 43 / 46 | 55.9 |
| V06 | 25 | 29 | 122 | 0.366 | 12 / 64 / 6 / 40 | 51.2 |
| V07 | 48 | 30 | 188 | 0.477 | 44 / 73 / 10 / 61 | 52.1 |
| V08 | 6 | 29 | 67 | 0.255 | 8 / 29 / 30 / 0 | 61.0 |

“PR 分数均值”仅平均该视频有定义的对象级 Path Relevance 分数；UNKNOWN 没有分数，不按 0 计入。V06 和 V08 各有一帧场景估计为 UNKNOWN。样本处理均值的计时从 `VideoCapture.read()` 返回采样帧后开始，到该帧对象关系、JSONL 和叠加 JPEG 写完为止；不含视频解码耗时，也不是端到端实时 FPS。

Shadow 建议按**采样目标/风险观察**合计：保留 Baseline HIGH 468、Path UNKNOWN 时保留 Baseline 219、保留高相关候选 67、人工复核 366、仅研究性降低优先级 3。它们不是 1,123 条不同提醒，也不是 257 个 AlertEvent 的过滤结果；生产决策改动数为 0。不得把 3 条研究建议解释为减少了 3 次误报。

三种来源的分布不同：A 中 1,123 个采样风险观察的平均 Baseline 风险分为 0.498；B 的几何 ROI 交叠均值为 0.102，217/1,123 个观察至少为 0.20；C 只对 741 个非 UNKNOWN 对象定义分数，均值为 0.319，其中 501 个至少为 0.20。B/C 分数定义和样本集合不同，不能把均值差当作性能优劣。分类型的 C 分数均值：person 0.307（n=371）、chair 0.611（n=116）、car 0.083（n=110）。类别主要改变接地点先验和置信度折减；这些分布主要反映本批次中目标所处的图像区域，不能解释为 chair 比 person 更危险或 car 更安全。

对照 A/B/C/D 的逐视频记录在 `outputs/path_relevance/v04/summary.json` 和每视频 `summary.json`；A 是原 Baseline，B 是固定中心下部 ROI，C 是 SegFormer 语义地面关系，D 是只记录建议的 Shadow。完整帧级和对象级详情分别在 `scene_frames.jsonl`、`objects.jsonl`、`shadow.jsonl`，叠加图在每段的 `overlays/`。

## 性能测量

- 模型加载：144.9 ms，一次加载供八段离线处理复用。
- SegFormer 前向推理：排除每段首帧后的 232 个样本平均 17.23 ms；同一组推理 p95 为 18.96 ms。240 个样本总集合的 p95 为 19.19 ms。V01 首个样本为 210.61 ms，主要包含首次 CUDA/模型热身，不计入 steady mean。
- 每视频样本处理均值范围：42.6–67.3 ms；按 8 个视频的均值再取平均约 54.0 ms。它含语义后处理、对象关联、JSONL 与 JPEG 叠加图写入，但计时从采样帧已经解码之后开始。
- PyTorch 报告的 CUDA peak allocated：196.11 MiB。它不是整卡占用或总进程显存；模型和其他进程的保留/分配显存不由此数覆盖。
- 批次墙钟约 19.0 秒（`started_at_utc` 至 `finished_at_utc`；GPU 模型已加载后运行八段）。不据此宣称解码吞吐或实时处理速度。

## 测试

自动化测试区分纯算法单元测试、可选模型集成测试与真实视频处理：

- `tests/test_path_relevance.py`：17 个本地单元测试覆盖走廊连通域、无区域 UNKNOWN、低质量弃权、类别接地点先验、重叠/区外关系、异常 bbox、分割尺寸对齐、时间戳、Shadow HIGH 保护和基线隔离。
- `tests/test_path_relevance_integration.py`：显式打开集成开关后读取真实本地 checkpoint 和 V05 UrbanEgo 原视频，并比较 Baseline alerts 哈希；GPU 环境结果为 1 passed，实际设备 CUDA。
- `uv run --no-sync ruff check .`：通过。`uv run --no-sync ruff format --check .`：169 个文件均已格式化。`uv run --no-sync pytest -q`：371 passed、6 skipped（22.82 s）。定向 `tests/test_path_relevance.py`：17 passed（1.76 s）。默认 skip 不计为真实推理通过；独立 V05 CUDA 集成测试为 1 passed、2 个 pytest 插件重写警告（2.92 s）。

## 复现

在本仓库根目录使用 CUDA 已验证环境。输出目录必须为空或不存在；本复现命令使用新目录 v05，避免覆盖已验证的 v04：

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import torch
from transformers import SegformerConfig, SegformerModel
sys.path.append(str(Path(".venv/Lib/site-packages").resolve()))
from sightguard.cli import main
sys.argv = [
    "sightguard", "path-relevance",
    "--config", "configs/path_relevance.yaml",
    "--manifest", "datasets/real_world/manifests/dataset.json",
    "--baseline-dir", "outputs/real_world/v01",
    "--output-dir", "outputs/path_relevance/v05",
]
raise SystemExit(main())
'@ | & "models/hybrid/env-structured-gpu/Scripts/python.exe" -
if ($LASTEXITCODE -ne 0) { throw "Path Relevance run failed" }
```

运行脚本先从专用解释器导入 CUDA Torch 和 Transformers，再追加工作区 `.venv` 依赖路径；若反过来把 `.venv` 放在 Python 搜索路径前面，会选中 CPU Torch。需要 CUDA Torch、Transformers、OpenCV、NumPy、PyYAML 以及本地 checkpoint。没有这些条件时可选 CPU，但性能结果会不同。可将 `--video-id urbanego_run8_05` 加入 `sys.argv` 单独复现 V05；八段结果使用同一份输入 tracks 和配置。

`outputs/path_relevance/v01/` 是低质量场景弃权修正前的探索批次；v02 是修正后较早 CUDA 批次；v03 使用修正后的许可元数据，但因 Python 搜索路径顺序意外加载了 CPU Torch；v04 在修正后的元数据和 CUDA 环境下重跑，是本报告引用的权威批次。每个目录均保留，不覆盖先前运行。批次目录均被 Git 忽略；视频、权重、overlay 和 JSONL 不随提交发布。

## 失败与解释边界

- V04 30 个采样场景中 27 个为 limited；108 个目标观察有 100 个 UNKNOWN、只有 8 个分数可用。该例说明算法会因视觉证据弱而弃权，但仍需标注验证其弃权条件是否合适。
- V05 的候选区域可把较宽的道路与人行道连在一起，因此与目标框的交叠较粗；V03 的框遮挡/目标框高度会影响接地关系。它们说明模型区域与 bbox 几何都可能不符合真实地面占用。
- V08 关键帧叠加看起来较连贯，但视觉合理不等于经过验证。
- 现有数据没有盲标候选地面边界、目标实际占道关系、运动中相机旋转或安全通行结果。无法回答这些实例究竟是误判、漏判还是正确关联，也无法报告 Accuracy、Precision、Recall、F1、误报率变化或安全收益。
