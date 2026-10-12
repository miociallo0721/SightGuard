> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Spatial Grid Fusion v1 验证报告

2026-10-11。本次完成独立 Shadow 网格、真实输入融合、同源消融、可视化和基线回归。
结果不构成空间占用准确率、碰撞预测或安全通行验证；目前没有独立空间占用真值。

## 审计与输入

基线 HEAD 为 `5291529998301f4f7a2162898fbc0d9f43ae2baf`，包含 Depth 实现
`c45f30057bf659a5f7a0ccccefb61de5a84cdc47`；新分支为
`codex/spatial-grid-fusion-v1`。AGENTS.md、全局配置、已有风险和提醒模块均未修改。
预先存在的未跟踪 `docs/architecture_consolidation_task_report_2026-10-11.md` 未纳入提交。

UrbanEgo 开发集八段视频，每段 15 秒、20 FPS、1280×720；30 个相同采样帧：
frame_index=0,10,…,290，timestamp_ms=0,500,…,14500。
manifest schema `sightguard.real-world.v0.1`，SHA-256：
`b4de54f10c7236968532792a05ad1997b4033f0f7c9ed9d5144421185bcb1d42`。
数据来自 [UrbanEgo 发布记录](https://zenodo.org/records/23062929)，CC-BY-4.0，
本地裁剪与复用说明沿用既有 dataset manifest；八段来自同一 parent source，不能视为八个独立样本群。

- YOLO/BoT-SORT：`outputs/real_world/v01/*/perception/tracks.jsonl`，未重新检测跟踪。
- Depth：`outputs/depth_integration/v03/`，复用 240 张真实 float32 源尺寸矩阵。
  checkpoint `depth-anything/Depth-Anything-V2-Small-hf`，revision
  `5426e4f0f36572d16453bbda7a8389317b1bef99`，SHA-256
  `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。
  本轮没有运行 Depth 模型。
- SegFormer：v04 原始目录没有 NPZ/原始 mask，只有 JPG、scene/object/shadow JSONL。
  因而用同一 `SegformerB0Perceptor` 与 `configs/path_relevance.yaml` 重新运行对应 240 帧，
  新来源为 `outputs/spatial_semantics/v01/`，不宣称这是 v04 原始 mask。
  OpenMMLab MiT-B0 ADE20K checkpoint SHA-256：
  `8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`。
  512 输入宽度，native shape=288×512；class、confidence、component 数组均无损保存。
  checkpoint 的独立许可验证沿用原阶段的未确认状态，本轮不分发权重。

SegFormer 240 帧真实推理均成功；238 帧有候选区域，2 帧 scene UNKNOWN（V06、V08）。
重新推理的 240 个 scene status 与 v04 完全相同；这仅核对有效性分布，不证明掩码与未保存的旧掩码一致。
GPU 集成测试另行核对 V05 首帧新推理 class map 与保存原生数组逐像素相等、confidence 在 1e-5 内一致。

## 实际输出与八段结果

最终代码对应批次：`outputs/spatial_grid/v03/`，240 帧、54,000 个单元、1,123 个对象关系。
`v01` 保存优化前探索批次，`v02` 保存优化后首批；最终 `v03` 的源文件 SHA 与当前交付代码一致。
v02/v03 全部 240 帧结构化结果逐行完全相同。均未覆盖旧目录。
语义重推理 metadata 的模型/估计器/runner 源码 SHA 也与交付代码一致。

每视频均有 30 帧、6,750 个单元、全部单元有至少一个有效非负深度像素。
以下顺序为 UNKNOWN / 框候选 / 地面 / MIXED；其和均为 6,750：

- V01：2,410 / 426 / 3,502 / 412；143 对象，711 个含冲突原因的单元。
- V02：3,068 / 1,232 / 1,759 / 691；211 对象，1,651 冲突单元。
- V03：2,727 / 1,723 / 1,940 / 360；144 对象，778 冲突单元。
- V04：2,822 / 3,263 / 389 / 276；108 对象，711 冲突单元。
- V05：2,878 / 983 / 2,645 / 244；140 对象，770 冲突单元。
- V06：3,572 / 1,188 / 1,475 / 515；122 对象，1,228 冲突单元。
- V07：3,296 / 925 / 1,864 / 665；188 对象，1,542 冲突单元。
- V08：2,265 / 407 / 3,855 / 223；67 对象，533 冲突单元。

冲突包含 MIXED 与相对深度变化诊断，重叠原因不重复计数。它不是错误率或 HIGH 风险。
225,082 个原深度负值像素被统计排除，占 221,184,000 个采样像素约 0.102%；
原始 NPZ 未修改。全部矩阵读取通过已有有限值校验，真实输入没有 NaN/Inf。
NaN/Inf 的部分统计与存储拒绝策略由独立数值/读取测试覆盖。

## 同源 A/B/C/D

四组都使用同一 240 个 FrameKey、源图尺寸与固定来源，排除来源不会被记录成读取异常。

- A（SegFormer）：20,815 地面、33,185 UNKNOWN；UNKNOWN 状态占 61.454%。
  不含深度/检测；没有对象关系。2 个语义 UNKNOWN 帧显式保留。
- B（Depth）：54,000 单元均有数值深度，状态仍全部 UNKNOWN。
  5,374 单元含深度变化原因，没有地面类别或检测占用证据。
- C（YOLO+SegFormer）：17,429 地面、10,147 框候选、3,386 MIXED、23,038 UNKNOWN；
  UNKNOWN 占 42.663%。1,123 对象几何关系，1,121 对象有可计算语义比例。
- D（三源）：状态分布与 C 相同，另有 54,000 单元和全部 1,123 对象的深度摘要。
  7,924 单元有一种或多种冲突原因，42,035 单元仍有未知/受限原因。
  2 个语义 UNKNOWN 帧没有被深度填成地面。

数值深度覆盖并不等于已知地面/占用。D 没有激进减少 C 的 UNKNOWN。
对象可计算几何比例为 100%，深度比例为 100%，语义比例为 99.822%；这些只是数值接口可定义性。
语义、相对深度、遮挡和 bbox 背景的真实性仍未被 GT 验证。
不计算 Occupancy Accuracy、F1、风险误报/漏报、规划成功率或安全性改善。

## 困难场景与保留样本

已逐图检查原始 RGB、地面、相对深度、检测、网格、底部候选对照；这不是独立人工占用标注。
记录在 `outputs/spatial_grid_validation/v02/manual_review.json`，所有 240 帧均保留，非仅展示最佳样本。

- V03 frame 200：6 个车类检测，132 UNKNOWN、48 框候选、41 地面、4 MIXED。
  大车框包含道路/遮挡背景，远车框覆盖较少；深度层次不能恢复被车挡住的地面。
- V04 frame 200：仅 10 格地面支持（其中 6 MIXED），215 格 ground.status=unknown，
  同时 225 格保留深度。窄人行区域与玻璃/近墙数值不能证明净空。
  整段 27 limited / 3 high scene，框占用证据不能替代语义。
  “更合理”指未知来源与有效数值分开可检查，不宣称地面判断准确性提高。
- V05 frame 290：9 个检测，15 MIXED、12 框候选。8 个 chair 框中地面支持比例约
  0.351–1.000，体现框内背景、椅腿间隙和语义误分的混合。
  实心 bbox 策略会有占用扩大候选，本实现保留覆盖比例及 MIXED，未证明已消除占用错误。
- V07 frame 100：8 对象、29 MIXED；人、桌椅、遮阳棚和招牌遮挡使区域证据混合。
  深度把遮阳棚表示为较近结构，不能直接认定它阻断地面。
- V08 frame 0：低矮环形台阶/柱体有数值结构，但小目标检测没有把这些位置标成真实障碍。
  深度变化仍是诊断；floor/road 识别不会自动发现台阶风险。
- V08 frame 290：相机俯向铺装地面，语义场景 UNKNOWN；225 格深度有效，
  225 格状态 UNKNOWN、没有地面补全。这是保留失败语义和未知姿态的样例。

## 性能和资源

硬件 Intel i7-12650H（10 核/16 逻辑处理器）、RTX 2050 Laptop 4,096 MiB，
NVIDIA driver 616.56；Windows 11。模型环境 Python 3.12.15、Torch 2.10.0+cu130、Transformers 5.19.0。
融合进程为 CPU NumPy/OpenCV，未加载 Torch/模型；以下 P50/P95 为 240 帧实测，单位 ms：

- RGB seek/decode：54.69 / 180.40。
- 输入总加载（含校验、解压、nearest 对齐）：44.35 / 59.79。
- 原 Depth NPZ hash/read/decode：30.15 / 35.48；不能称为 Depth 模型实时推理。
- 原生语义读取/hash/解压/对齐：14.01 / 24.87。
- mask nearest 对齐子步骤：1.86 / 4.57，238 个有效语义帧；Depth 已对齐源图，无重采样。
- 单元聚合：81.78 / 108.92。
- 对象覆盖与框内统计：5.65 / 19.68。
- 总融合（含全帧准备、排序、上述步骤）：127.84 / 155.34。
- JSONL 转换/序列化/写入：9.79 / 14.09。
- 六面板和网格 JPG、交互数据 JS 保存：107.97 / 174.89。
- 完整 D 采样帧（含 decode/load/fusion/JSONL/visualization）：363.02 / 535.78。

输入/对齐计时是嵌套的，不能相加；各 P50/P95 不能相加还原总体分位数。
A/B/C 纯融合 P50/P95 分别为 48.41/59.60、69.96/83.62、51.24/66.80；
D 以 `fusion_total` 为准。metadata 中 `ablation_D` 是复用已计算 D 的选择耗时，
**不能拿它的约 0.145ms 当 D 融合耗时**。
批次墙钟 119.20 秒，包含 A/B/C 补充计算、批次元数据和输出，模型重推理单独运行。
融合进程结束工作集 83.57 MiB，进程峰值工作集 207.53 MiB（Windows GetProcessMemoryInfo）。
这是整个该进程峰值，不是 NumPy 数组净内存，也不含另一个模型进程或其他服务。
测量时保留用户已有服务与探索批次；不是独占机器微基准，不保证实时帧率。

SegFormer 单独批次墙钟 45.52 秒、冷加载 3,648.70ms、2 次预热。
同步 CUDA forward/softmax 等适配器推理 P50/P95=21.83/25.10ms，
完整 predict（含预处理、CPU 数组导出）34.44/41.04ms，完整语义采样含读取/保存为 159.08/254.38ms。
第一条完整语义采样还包含预热；模型推理计时均在预热之后。
Torch peak allocated=196.11 MiB、peak reserved=306.00 MiB；不是 GPU 全设备峰值。
模型进程退出后 `nvidia-smi` 显示显存使用 0 MiB。本轮没有 CPU SegFormer 吞吐测量；不能外推。

## 验证、兼容性与限制

- 默认：`uv run --no-sync pytest -q` → **457 passed、16 skipped，29.56s**。
  新增纯算法测试 54 项，覆盖配置、分块、精确身份、来源部分失效、错误检测、数值、混合、
  确定性、溯源、跨视频隔离、写入失败、拒绝覆盖和不依赖决策/音频/模型。
- 真实保存输入：6 passed，包含 V03/V04/V05/V07/V08 数值逐格校验及真实单帧 pipeline。
- 实际 GPU SegFormer：1 passed；与原生缓存核对 class/confidence。
  第一次该测试遗漏显式 UTF-8，Windows GBK 读取 manifest 失败；修正读取编码后实际推理通过。
- Ruff check 通过，最终 format --check：202 files already formatted。
- 冻结文件 890 个 SHA 前后相同，涵盖 Depth v03、Path v04、real_world v01、
  Phase 3 final/development、AGENTS、uv.lock 和既有未跟踪报告。
- 八段 2,400 帧独立默认风险重放，risk_frames/events/alerts JSONL 与原结果逐行一致；
  AlertEvent 总计 257 条，各段 33/55/34/20/36/25/48/6。没有把网格接入正式调度，没有音频调用。
- 最终融合与语义 metadata 的源代码 SHA 均与交付源码一致。
- 正常 `uv sync --locked --extra depth --extra path-relevance` 解析 110 个包，
  但因运行中的 `sightguard.exe` 文件锁报 os error 32，尚未验证无文件锁条件下的正常同步成功。
  未强制结束服务，uv.lock/依赖保持不变。

六面板 JPG 已实际查看；HTML 保存 240 帧入口、单元和对象的溯源数据。
Node VM 的 DOM 模拟验证选帧、225 单元、边缘像素点击、对象切换和过时帧响应保护均通过。
浏览器工具因 URL 策略拒绝本地 file://，未绕过限制；真实浏览器渲染未验证，
不把 Node 模拟当浏览器 E2E。可手动打开最终 `index.html` 查看。

最大限制是单目相对尺度、未观测相机姿态、低分辨率语义、框内背景与遮挡，
以及尚无独立空间真值；三源均成功推理也不构成可靠物理空间判断。
下一阶段优先复用严格 FrameKey、局部 source_status/quality/reasons、组件联系、
ObjectGridRelation 与原始文件引用，先建立独立标注和候选走廊评价，不能省略未知传播。

## 复现

以下从仓库根目录运行；输出需选择不存在的新版本名，已有批次不可覆盖。
复用语义缓存的融合不需要 GPU，也不需要加载 Depth/SegFormer 模型：

```powershell
uv run --no-sync python scripts/run_spatial_grid.py --output outputs/spatial_grid/new-run
uv run --no-sync python scripts/validate_spatial_grid.py --output outputs/spatial_grid_validation/new-run --fusion outputs/spatial_grid/new-run
$env:SIGHTGUARD_RUN_SPATIAL_SAVED_INTEGRATION='1'
uv run --no-sync pytest -q tests/test_spatial_grid_integration.py -k 'not actual_segformer'
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

在带本地 checkpoint 和 Transformers 的 CPU/CUDA 环境重建语义：

```powershell
uv run --no-sync python scripts/prepare_spatial_semantics.py --output outputs/spatial_semantics/new-run --device cpu
```

本机 CUDA 隔离环境实际命令（先导入 GPU Torch/Transformers，后补充主环境 OpenCV 等依赖）：

```powershell
& 'models/hybrid/env-structured-gpu/Scripts/python.exe' -c "import sys,runpy,torch,transformers;sys.path.insert(0,'src');sys.path.append('.venv/Lib/site-packages');sys.argv=['scripts/prepare_spatial_semantics.py','--output','outputs/spatial_semantics/new-run','--device','cuda:0'];runpy.run_path(sys.argv[0],run_name='__main__')"
$env:SIGHTGUARD_RUN_SPATIAL_MODEL_INTEGRATION='1'
& 'models/hybrid/env-structured-gpu/Scripts/python.exe' -c "import sys,torch,transformers;sys.path.insert(0,'src');sys.path.append('.venv/Lib/site-packages');import pytest;raise SystemExit(pytest.main(['-q','tests/test_spatial_grid_integration.py','-k','actual_segformer']))"
```

改用新语义缓存时给融合命令传 `--semantics outputs/spatial_semantics/new-run`。
如需重新建立冻结文件基线，运行前对相同旧输入文件做 SHA 快照并传验证入口的 `--frozen`。
模型、视频、真实结果与生成图片都位于 Git 忽略目录，不提交或下载。

实现细节和上游固定版本引用见 [spatial_grid_fusion_v1.md](../research/spatial_grid_fusion_v1.md)。
