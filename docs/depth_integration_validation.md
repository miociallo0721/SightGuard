# Depth Integration v1 验证记录

日期：2026-10-11。真实模型/真实视频与 Mock 单元测试分别报告。
**完成的是相对深度数据集成；没有验证导航、真实距离、风险准确率或安全通行收益。**

## 保护与交付范围

起点 `9059aa8002a1e93554ea86f973b9089b9abc3b8a`，分支
`codex/architecture-consolidation-v1`；Path Relevance `3902563` 存在。
本地领先 origin/master 12 提交；`git ls-remote` 的远程 master 仍为
`757468ec4ef5eb4dea23fbd129e0971d87dee28a`。从当前 HEAD 创建
`codex/depth-integration-v1`，不合并、不 reset、不 clean、不 push。
原未跟踪 `docs/architecture_consolidation_task_report_2026-10-11.md` 保留且不纳入本次提交。
AGENTS.md、原有算法源码、所有风险配置、GT 3.0、冻结协议、已有实验产物未改。

新增四个 perception/depth 文件、独立 pipeline/depth、depth YAML、prepare/run/validate
三脚本、两份测试、两份文档。README 更新索引；pyproject.toml/uv.lock 增加共用 Transformers
的 depth extra，没有修改既有包解析版本。详见[模型接口文档](depth_integration_v1.md)。

## 实际环境与模型

- Windows 11 build 26200、Python 3.12.15。
- RTX 2050 Laptop、4096 MiB、NVIDIA driver 616.56。
- 隔离 `models/hybrid/env-structured-gpu/Scripts/python.exe`：Torch 2.10.0+cu130、
  Transformers 5.19.0、NumPy 2.5.3、OpenCV 4.14.0.94、safetensors 0.8.0、Pillow 12.3.0。
- 官方 Depth Anything V2 Small HF revision `5426e4f0f36572d16453bbda7a8389317b1bef99`，
  checkpoint hash `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`，
  实际加载 24,785,089 参数，loading_info 无 missing/unexpected/mismatched keys。
- CPU 与 CUDA 均真正执行 forward；没有静态图片预测、随机模型、Mock 替代真实批处理。
  模型来源与 Apache-2.0 核实链接见接口文档。

## 数据来源与完整批次

输入 manifest SHA-256：`b4de54f10c7236968532792a05ad1997b4033f0f7c9ed9d5144421185bcb1d42`。
UrbanEgo run8 同一原始拍摄，development，CC-BY-4.0；[来源记录](https://zenodo.org/records/23062929)
和逐视频 provenance 沿用已存在的 manifest，未重新选择片段。每段 1280×720、20 FPS、300 帧，
完整解码 8×300 帧；每 0.5 秒采样，保存各 30 帧，共 240 份 NPZ/JPEG。

**权威最终批次 `outputs/depth_integration/v03/`**，240 valid、0 UNKNOWN、0 推理失败。
此处 valid 仅为工程检查通过。每视频帧索引 0,10,…,290，时间戳 0,500,…,14500 ms，
与既有 tracks、risk_frames、Path Relevance v04 严格相同；depth 为 720×1280，原生数组为
280×504，均 float32。各段输入 SHA-256：

- V01：`7d458c438ecb8eb502f08fd55368f0500fe0ea46bdeae0cd24a3a737bda9f40f`
- V02：`20397fc566e7dcf78c2ed43b7081095df98bd8a2490ad62c84879e6f3308a251`
- V03：`a88348f206a45a11fcc563d2a2f3143db7f0df9e7b0266725c54e62a23c9c769`
- V04：`22ead9e431b2a4f8343ba87bff9d9651e871ec29671d8e7687762db10d6f4dc7`
- V05：`806b7851b76e6cde4fb3559053afbab8f637449faf7a25aa39291c1f0a22684f`
- V06：`8bee0ce37da9dd6fe255ea4581871200e6f82c34e0c9c4be8cdde053315d8cdb`
- V07：`265a726bba94fa494834dc56af7f135bf611b4ef8337d3a4d08034740ce03aa4`
- V08：`bf5c37c5eb356a897bae55e9d407d661cfa5c24e7088d37e718f241d526dc43c`

探索记录保留：`outputs/depth_probe/` 是单帧最小真实推理；v01 因 tracks 字段错误和 HF set
序列化问题失败，视频 summary 保留但 batch metadata 未成功写出，不计成功。修正后的 v02
为 240/240，墙钟 88.95 s；最终 v03 使用最终 adapter/runner，墙钟 70.67 s。
差异反映系统负载与 I/O，不能当成模型优化收益。所有目录均未覆盖。

## 近远方向数值验证

真实 V05 frame0 原 RGB 可见近处右墙和远处停放车辆。使用原图坐标的固定区域：
近墙 `[y=200:500, x=1100:1200]`，远车 `[y=50:100, x=520:660]`。
CPU 预测中位数分别 7.73920 / 0.200002；CUDA 分别 7.73929 / 0.199959。
这支持该 checkpoint 的 **larger = nearer** 方向，不依赖热力图色彩；不构成度量深度标定或
所有物体排序准确率验证。没有把未归一化值求倒数解释为米。
原生预测非负；双三次空间恢复有少量负值，保留原值与原生数组，不裁剪。

## 人工检查与困难案例

检查 v02 的各视频 frame0/100/200/290 RGB/深度对照，v03 各帧也保留完整预览。
最终八份拼图由 `scripts/validate_depth.py` 生成在 `outputs/depth_validation/v02/`。
这是代理人工视觉检查，不是独立盲标 GT；下述现象不能用来报告 Precision/Recall 或导航收益。

- **V03**：frame0 近菜单牌、车辆和背景有层次；frame100 摩托车/覆盖物与道路、远车有分离。
  frame200/290 车窗、反光/遮挡边界及远处小车较平滑，部分轮廓丢失，无法可靠判断细物体占用。
- **V04**：frame0/100 近车、近墙、远车出现不同层次，与 SegFormer 的地面类别证据不同。
  原 Path 场景 limited/对象 UNKNOWN 不被此处 valid 覆盖；玻璃反射与窄路边缘仍难以验证，
  不能因为深度数值有效就把 100 个对象 UNKNOWN 改成可通行。
- **V05**：frame0 近墙显著大于远车；frame100 道路车辆和近墙分层；frame200/290
  咖啡座、桌椅、人和背景有粗层次，但细腿、桌间间隙、暗区轮廓不清。没有检测框风险融合。
- **V07**：frame100/200 多人、桌椅、招牌与遮阳棚有层次；棚布很近并占据显示动态范围，
  人/椅细节可能被压缩，遮挡及细腿难判。这些真实困难案例保留，不只展示较好结果。
- **V08**：frame0 纪念物/底座、frame200 长椅与柱子分层；frame100/290 大地面渐变看似平滑。
  frame0 的低矮台阶高度与危险性未被验证，低提醒密度不表示安全对照。

240 帧未触发 UNKNOWN 只能说明本批次未碰到所定义的工程弃权阈值，不代表模型没有预测错误。
没有真实 NaN/OOM 故障时强造故障；这些异常由 Mock 路径测试覆盖，明确与真实模型分开。

## CUDA 性能与资源

v03 独立 CUDA FP32、模型复用，input 504×280。初始 load **2119.51 ms**，包括哈希校验、
optional runtime 导入、权重加载、设备转移和同步。两次真实 frame0 热身单独记入 runtime，
不计下列 240 个采样推理分位数。部分验证作业并行运行，数字仅代表这次本机测量。

- 预处理 P50/P95：12.95 / 15.11 ms。
- forward P50/P95：37.58 / 37.90 ms。
- 后处理/CPU 回传 P50/P95：4.79 / 5.23 ms。
- 总推理 P50/P95：55.21 / 59.11 ms。
- NPZ 压缩写入+hash P50/P95：183.05 / 196.41 ms。
- JPEG 可视化保存 P50/P95：9.88 / 12.38 ms。
- 单采样完整处理 P50/P95：260.60 / 283.63 ms；包含选中帧 read、推理/质量与保存，
  不含非采样帧解码、热身和 JSONL 写入。batch wall 70.67 s 包含完整解码/热身/元数据。

每视频推理 P50/P95（ms）：V01 56.59/58.81；V02 56.15/57.63；V03 56.57/69.76；
V04 55.12/80.44；V05 54.95/55.74；V06 55.07/55.89；V07 55.14/56.01；V08 54.88/56.04。
这些不是实时系统 FPS，不能推算多模型 Live 系统达到 20 FPS。

PyTorch peak allocated **162.04 MiB**、peak reserved **202.00 MiB**；运行结束卸载前
allocated 103.68 MiB。nvidia-smi 抽样整卡运行占用 **287 MiB / 4096 MiB**，不声称这是整批
整卡峰值；WDDM 下未取得可靠的单进程显存统计。GPU 前/后快照 0/107 MiB。
close 后 allocated 9.125 MiB、reserved 22 MiB，模型对象 weakref 已失效，权重分配释放
约 94.555 MiB。残留为 PyTorch/cuBLAS 工作区；在独立诊断进程调用私有清理接口后
allocated/reserved 均为 0（`outputs/depth_stage0/unload_diagnosis.json`），适配器不使用这个
全进程清理接口，也不声称 close 后 CUDA context 不占显存。

## CPU 与短时稳定性

`outputs/depth_validation/v01/validation.json` 另存真实单帧重复 benchmark：相同 V05 frame0、
FP32、2 热身、CPU 10 次、CUDA 60 次，10 个 CPU threads，无解码/保存。

- CPU preprocess 16.16/18.62、forward 141.68/148.89、postprocess 4.16/4.66、
  total 162.89/169.88 ms（P50/P95）。模型 load 2067.67 ms。
- CUDA total 50.95/54.06 ms、forward 32.95/36.04 ms；同进程已经导入运行库后 load 316.68 ms，
  不与冷启动 load 直接比较。
- CUDA 60 次 allocated 全部 103.680 MiB，无持续增长；无 OOM。这只验证数秒重复单帧的
  短时稳定性，不证明长时间、分辨率变化或多模型常驻时不会 OOM。

CPU 模式真实可用；未在 CPU 重跑全部 240 帧，八段完整实验使用 CUDA。未测试真实 OOM，
受控 fallback 单元测试成功；不假称真实 OOM 恢复已经验证。

## 回归与测试

起点默认 pytest：371 passed、6 skipped。新增 32 个 Mock/本地 I/O/契约单元测试，覆盖
配置与 hash、缺权重、模拟 checkpoint 加载/后端选择/只加载一次、颜色、尺寸、NaN/Inf、
平坦 UNKNOWN、失败不复用旧结果、一次 OOM fallback、无损存储、显示与数据隔离、读取
hash/路径防护、确切帧时间与跨视频隔离、坏视频、拒绝覆盖及导入不激活决策/音频。

- `uv run --no-sync ruff check .`：通过。
- `uv run --no-sync ruff format --check .`：最终包含文档示例的 186 个文件均格式正确。
- `uv run --no-sync pytest -q`：**403 passed、9 skipped，26.18 s**。
  6 个既有 opt-in skip 保留，新增 3 个真实 depth opt-in 默认 skip；skip 不计集成通过。
- 显式 `SIGHTGUARD_RUN_DEPTH_INTEGRATION=1`，GPU 隔离解释器实际执行
  `tests/test_depth_integration.py`：**3 passed、2 插件重写 warnings，14.36 s**。
  包含真实 CPU、真实 CUDA、1280×720/640×360 尺寸与方向测试，以及 V05 全 30 帧真实 E2E。
- 早期 CUDA 测试要求 close 后 allocated <1 MiB，发现 cuBLAS 残留后失败；经真实诊断确认
  这不是模型权重泄漏，改为同时检查模型弱引用失效及完整权重分配释放，没有隐藏失败。
- 请求中的不带 `--no-sync` 三个 uv 命令均实际尝试，但同步 editable package 时被正在使用的
  `.venv/Scripts/sightguard.exe` 锁阻止（os error 32）。未强停用户进程；同一现有环境中
  `--no-sync` 检查通过。待该进程正常退出后可再执行同步版本。

真实八段 tracks 无模型重放：2,400 帧、257 AlertEvent，各段提醒 33/55/34/20/36/25/48/6；
risk_frames、events、alerts **逐条与历史 baseline 相等**。输入 Baseline 与 Path v04 的
84 个历史产物 hash 在前后校验一致。此证明默认行为兼容，不证明风险判断正确。
没有改既有测试断言/skip、冻结实验协议或生产风险/音频代码。

## 实际命令和证据位置

```powershell
& models/hybrid/env-structured-gpu/Scripts/python.exe scripts/run_depth.py --workspace-dependencies --device cuda:0 --output-dir outputs/depth_integration/v03
& models/hybrid/env-structured-gpu/Scripts/python.exe scripts/validate_depth.py --workspace-dependencies --run-dir outputs/depth_integration/v02 --output-dir outputs/depth_validation/v01 --benchmarks
uv run --no-sync python scripts/validate_depth.py --run-dir outputs/depth_integration/v03 --output-dir outputs/depth_validation/v02 --artifact-baseline outputs/depth_stage0/artifacts.json
```

这些目录已经存在，复现必须换新目录。真实集成测试运行方式：

```powershell
$env:SIGHTGUARD_RUN_DEPTH_INTEGRATION = "1"
@'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import torch, transformers
sys.path.append(str(Path(".venv/Lib/site-packages").resolve()))
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_depth_integration.py"]))
'@ | & models/hybrid/env-structured-gpu/Scripts/python.exe -
```

主要证据为 v03 run_metadata、各视频 depth_frames/alignment/summary/NPZ；
`outputs/depth_validation/v02/validation.json` 检查完整数值/hash/时间并重放 baseline，
v01 validation 保存 CPU/CUDA benchmark。未把视频、权重、矩阵、拼图、大日志提交 Git。

## 尚未解决

没有独立相对深度/遮挡/空间占用标注，没有米制标定、相机姿态、跨帧尺度一致性、像素可靠性，
无法验证桌椅细腿、透明反光、低矮台阶、远处小物体。没有多模型并发资源测试、长时间内存稳定性
或真实 OOM 恢复验证。下一阶段可读取本契约和原始矩阵/关联索引，但必须继续保留 UNKNOWN，
单独验证网格与证据融合。本轮到此停止，不自动进入 Spatial Grid Fusion。
