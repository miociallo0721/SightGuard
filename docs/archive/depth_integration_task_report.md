> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard — Depth Integration v1 任务报告

日期：2026-10-11。项目：SightGuard（视界）。

实现分支：`codex/depth-integration-v1`。
实现提交：`c45f30057bf659a5f7a0ccccefb61de5a84cdc47`。
提交说明：`experiment: add monocular relative depth perception`。

## 一、交付结论

本阶段完成了独立、真实、可追溯的 Depth Anything V2 Small 相对深度感知集成。
模型已在本机 CPU 和 RTX 2050 CUDA 环境实际执行 forward；UrbanEgo 八段视频完整解码，
按 0.5 秒间隔生成 240 个采样帧的深度矩阵、元数据、关联索引和可视化。
最终批次全部通过工程有效性检查，输入来源、模型版本、权重哈希和视频时间可以追溯。

默认风险与提醒系统未接入深度结果。重放原有 2,400 帧得到的 risk_frames、events、alerts
与历史结果逐条一致，257 条正式 AlertEvent 保持不变。

该交付证明模型推理、数据保存和接口兼容性已经跑通。它没有证明深度排序在所有场景中准确，
没有验证米制距离、可通行空间、真实碰撞预测或视障使用者安全收益。

## 二、任务范围与工作保护

开发起点为架构整理提交 `9059aa8002a1e93554ea86f973b9089b9abc3b8a`，
已有 Path Relevance 集成提交 `3902563`。检查时本地领先 origin/master 12 个提交，
从当前成果建立独立深度开发分支，没有切回旧 master 或合并远程分支。

本次实施模型核实、真实单帧推理、Depth Adapter、离线 Runner、八段实验、
性能和资源验证、单元及集成测试、原提醒回归和文档交付。

保护措施与边界：

- AGENTS.md、原有生产算法、风险配置、Ground Truth Schema 3.0 和冻结实验协议保持不变。
- 不覆盖未提交文件，不删除历史实验，不使用 reset --hard、git clean 或强制推送。
- 原未跟踪的架构任务报告保留，不纳入深度实现或本报告的提交。
- 未实施 Spatial Grid Fusion、Corridor Planner、动态策略、反馈系统或移动端移植。
- 未调整 Risk Score、HIGH 优先级、AlertScheduler、HRTF 或 Laya Shadow。

新增源码会改变既有全源码 benchmark fingerprint；旧冻结协议与历史结果保留，没有自动重封。

## 三、实际模型、权重与运行环境

模型为官方 `depth-anything/Depth-Anything-V2-Small-hf`，DINOv2 ViT-S + DPT，
使用 Transformers 的 `DepthAnythingForDepthEstimation`，实际参数数为 **24,785,089**，FP32。

- 固定 revision：`5426e4f0f36572d16453bbda7a8389317b1bef99`。
- 权重文件：`model.safetensors`，99,173,660 bytes。
- 权重 SHA-256：`3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。
- 配置、预处理配置及 model card 的独立哈希也已记录并校验。
- 官方源码许可与 Small checkpoint 许可核实为 Apache-2.0；依赖与数据沿用各自许可。
- 权重保存在被 Git 忽略的模型目录，不随本次提交分发。运行时只加载本地文件，不隐式下载。

实际环境为 Windows 11、Python 3.12.15、RTX 2050 Laptop 4 GiB、NVIDIA driver 616.56。
GPU 研究复用已有隔离解释器，其 Torch 为 **2.10.0+cu130**、Transformers 为 **5.19.0**；
NumPy 2.5.3、OpenCV 4.14.0.94、safetensors 0.8.0、Pillow 12.3.0。
它与主 `.venv` 的 Torch 环境不同，报告没有混用两者的验证结果。

`depth` optional extra 复用已有 Transformers 依赖，uv.lock 只增加 extra 映射，
没有改变既有解析版本。完整来源和许可链接见[模型与接口文档](../research/depth_integration_v1.md)。

## 四、实现与文件交付

实现提交包含 16 个文件：

- `src/sightguard/perception/depth/__init__.py`：独立深度模块导出。
- `src/sightguard/perception/depth/schema.py`：DepthModelConfig、DepthFrameResult 与版本化契约。
- `src/sightguard/perception/depth/model.py`：本地权重校验、加载、设备选择、真实推理、质量检查和卸载。
- `src/sightguard/perception/depth/artifacts.py`：无损矩阵保存、独立显示、校验与安全读取。
- `src/sightguard/pipeline/depth.py`：manifest 校验、真实视频解码、确定性采样、时间关联与批次记录。
- `configs/depth.yaml`：模型版本/hash、设备、输入尺寸、采样、热身和质量参数。
- `scripts/prepare_depth.py`：显式准备固定版本模型，校验文件大小与哈希。
- `scripts/run_depth.py`：独立深度 CLI，不进入原默认提醒链路。
- `scripts/validate_depth.py`：数组/时间校验、对照拼图、原风险重放及可选 CPU/CUDA benchmark。
- `tests/test_depth.py`：32 个本地单元测试，模拟模型与异常路径明确标识。
- `tests/test_depth_integration.py`：显式开启的真实 CPU、CUDA、UrbanEgo E2E 测试。
- `docs/research/depth_integration_v1.md`：模型、接口、语义、数据契约与复现说明。
- `docs/archive/depth_integration_validation.md`：真实环境、实验、性能、失败与测试证据。
- `README.md`：增加独立深度入口和当前阶段索引。
- `pyproject.toml`、`uv.lock`：增加共用 Transformers 的 depth extra。

本任务报告是上述实现提交之后补充的文档，不改变运行代码。

## 五、Depth Adapter 与数据契约

DepthEstimator 支持显式 initialize、逐帧 infer、close。模块导入不会加载模型，
initialize 不逐帧重复执行；CPU/CUDA 的实际设备会记录到结果。
缺权重、hash 不匹配或 CUDA 不可用具有明确错误；CUDA OOM 可按配置进行一次 CPU fallback，
其他错误不会无限重试或复用上一帧深度。

输入为原始 uint8 BGR，转换为独立 RGB 数组。固定 DPT 预处理保留长宽比、采用 14 的倍数，
执行 rescale 与 ImageNet normalization。配置 input_size=280 在本批次实际生成
**504×280** 模型输入；原图为 **1280×720**。

直接取得模型的未归一化 predicted_depth，同时保留：

- 原生 float32 数组：shape 为 280×504，按 `[y, x]` 索引。
- 双三次恢复到原图的 float32 数组：shape 为 720×1280，可与原图像素坐标关联。
- 仅用于展示的每帧归一化热力图，不作为后续算法输入。

输出使用 `sightguard.depth.v1`，保留既有 video_id、stream_id、frame_index、timestamp_ms
及时间来源，并记录模型/revision/hash、device、尺寸、数值统计、状态、失败原因、
性能与矩阵引用。没有改写原 FrameRecord、Detection 或 AlertEvent。

status 区分 valid/unknown；inference_status 区分 model_unavailable、image_read_failed、
inference_failed 和 completed。NaN/Inf、无效尺寸及缺乏有效变化会被拒绝或标为 UNKNOWN。
没有经过标定的置信度；valid 仅表示工程检查通过，UNKNOWN 不能转成“安全”或“无障碍”。

## 六、相对深度语义验证

本 checkpoint 的数值为相对逆深度式表示，**较大值表示画面内相对更近，较小值表示相对更远**。
方向验证使用真实 V05 第 0 帧原始 RGB 中的近墙和远车区域，而不是根据热力图颜色猜测。

近墙区域 `[y=200:500, x=1100:1200]`、远车区域 `[y=50:100, x=520:660]` 的预测中位数：

- CPU：近墙 7.73920，远车 0.200002。
- CUDA：近墙 7.73929，远车 0.199959。

这是一项近远方向合理性检查，不是独立深度真值评价。没有把数值称为米，
没有假定跨帧/跨视频尺度一致，没有计算真实速度、TTC 或米制危险阈值。
双三次恢复产生的少量负值原样保存，原生矩阵也同时保留。

## 七、八段真实实验与输出

权威批次为 `outputs/depth_integration/v03/`。输入来自 UrbanEgo run8 同一次原始拍摄，
八段均为 development，每段 15 秒、20 FPS、300 帧，合计完整解码 2,400 帧。
视频文件哈希逐一与 manifest 匹配，未重新选择片段。

V01、V02、V03、V04、V05、V06、V07、V08 **各保存 30 个有效采样结果、0 个 UNKNOWN**，
合计 240 valid、0 推理失败。240 份保存矩阵的哈希、dtype、尺寸及数值均完成独立读取校验。
这个计数不等于 240 次空间判断全部正确。

采样帧为 0、10、…、290，时间为 0、500、…、14500 ms。
时间读取既有 tracks，核对 risk_frames、Path Relevance v04 与实际解码 PTS，
不使用推理完成时间。原始传感器 capture time 与 MP4 的映射仍未获得，不作推断。

输出布局为：

- 每视频 `depth_frames.jsonl`：帧级元数据及 NPZ/JPEG 引用。
- `depth/000000.npz`：`depth` 为原图对齐数组，`model_output` 为原生数组，无损压缩。
- `previews/000000.jpg`：原图与热力图、时间、状态、尺寸和统计。
- `alignment.jsonl`：tracks/risk/scene 行索引与原 event IDs，仅关联，不融合风险。
- `summary.json`、`anomalies.json`：诊断、失败与 UNKNOWN 索引。
- 批次 `run_metadata.json`：来源、配置、版本、源码哈希、硬件、热身、资源和时间记录。

`outputs/depth_validation/v02/` 保存八段对照拼图、完整结果校验、baseline 重放证据和困难案例索引。
模型、视频、矩阵、可视化及大型日志均保留本地，不提交 Git。

## 八、重点场景观察与失败记录

这些观察来自代理对 RGB/深度拼图的视觉检查，不是独立盲标或真实空间测量。

- V03：菜单牌、车辆与道路背景可见粗层次；反光车窗、遮挡边界和远处小车较平滑，细节可能丢失。
- V04：近车、近墙和远车提供了不同于语义分割的空间证据；玻璃反射与窄路边界仍难判。
  深度 valid 没有覆盖原 Path Relevance 的 limited 或对象 UNKNOWN。
- V05：近墙与远车分层明显，咖啡座桌椅有粗层次；细腿、桌间间隙与暗区轮廓没有得到可靠验证。
- V07：多人、桌椅、招牌与遮阳棚存在层次；近棚布占据显示范围，密集遮挡与细物体难以判读。
- V08：长椅、柱子与纪念物有分层；低矮台阶的高度与危险性未验证，低提醒密度不表示安全。

早期 v01 出现 tracks 尺寸字段误用与 HF set 序列化错误：视频 summary 保留，
batch metadata 未成功写出，不计成功。修正后的 v02 完成 240 帧，最终 v03 使用最终 adapter/runner。
另一次 CUDA 卸载测试发现 cuBLAS 工作区残留，经诊断区分为进程级库资源，模型权重已释放。
所有探索产物保留，没有删除失败记录或仅挑选效果好的图像。

## 九、性能与资源

v03 使用独立 CUDA FP32 模型，模型加载与初始化耗时 **2119.51 ms**，两次热身单独记录。
以下为 240 个采样帧的 P50/P95：

- 预处理：12.95 / 15.11 ms。
- 模型 forward：37.58 / 37.90 ms。
- 后处理和 CPU 回传：4.79 / 5.23 ms。
- 总推理：55.21 / 59.11 ms。
- NPZ 压缩保存及 hash：183.05 / 196.41 ms。
- 可视化保存：9.88 / 12.38 ms。
- 单采样完整处理：260.60 / 283.63 ms。

批次总墙钟 **70.67 秒**。单采样计时不含非采样帧解码、热身和 JSONL 写入；
批次墙钟包含完整流程。CUDA 计时进行了同步，不能把单模型推理速度解释为实时系统 FPS。
结果显示本轮磁盘压缩保存比模型 forward 更耗时，但不同批次墙钟差异不能直接当作算法优化收益。

PyTorch peak allocated **162.04 MiB**、peak reserved **202 MiB**。
nvidia-smi 的运行抽样为整卡 **287 MiB / 4096 MiB**，不是整批峰值或单进程显存。
卸载后模型对象释放，完整权重分配回收；allocated 9.125 MiB、reserved 22 MiB 为
库工作区残留，独立诊断清理 cuBLAS 后降为 0。适配器没有使用全进程私有清理接口。

另一次真实 V05 单帧重复 benchmark 使用 2 次热身、CPU 10 次、CUDA 60 次：

- CPU 总推理 P50/P95：162.89 / 169.88 ms，10 个 CPU threads。
- CUDA 总推理 P50/P95：50.95 / 54.06 ms。
- CUDA 60 次 allocated 均为 103.680 MiB，没有持续增长或 OOM。

CPU 真实模式可用，但未在 CPU 重跑八段完整批次。稳定性实验只有数秒单帧重复，
未证明长时间、多模型常驻或不同输入分辨率下不会 OOM。真实 OOM fallback 未验证，
一次受控重试路径由 Mock 测试覆盖。

## 十、测试与兼容性证据

开发前默认测试为 371 passed、6 skipped；完成后为 **403 passed、9 skipped**。
新增 32 个单元测试，覆盖配置、权重、加载与后端、颜色与尺寸、异常输出、状态与时间隔离、
保存与可视化分离、读取校验、拒绝覆盖、视频失败以及不激活决策或音频。

历史检查结果：

- `uv run --no-sync ruff check .`：通过。
- `uv run --no-sync ruff format --check .`：实现交付时 186 个文件格式正确。
- `uv run --no-sync pytest -q`：403 passed、9 skipped，26.18 秒。
- 显式真实模型集成：3 passed、2 个插件重写 warnings，14.36 秒；包含 CPU、CUDA 和 V05 全 30 帧 E2E。

默认 skip 是 opt-in 真实模型测试，未计作推理成功；真实集成结果来自单独开启后的实际运行。
没有删除、跳过或弱化既有有效测试。报告编写阶段只核对已有证据，不声称重新执行了模型实验。

需求中的不带 `--no-sync` 命令曾实际尝试，但 uv 同步时遇到正在使用的 sightguard.exe 文件锁。
未强停用户进程；检查改用已安装环境的 --no-sync。同步版本仍需在该进程正常退出后复核。

原有八段 tracks 重放的正式提醒分别为 **33、55、34、20、36、25、48、6**，共 257 条。
全部 risk_frames、events、alerts 与历史 baseline 逐条相同，84 个既有研究产物哈希不变。
证明默认行为兼容，没有证明提醒本身的风险准确性。

## 十一、调用与交接方式

从仓库根目录，在已有 CUDA 隔离环境运行以下命令；输出必须选尚不存在的新目录：

```powershell
& models/hybrid/env-structured-gpu/Scripts/python.exe scripts/run_depth.py --workspace-dependencies --device cuda:0 --output-dir outputs/depth_integration/new_run
uv run --no-sync python scripts/validate_depth.py --run-dir outputs/depth_integration/new_run --output-dir outputs/depth_validation/new_validation
```

`--device cpu` 可以选择 CPU，`--video-id urbanego_run8_05` 可以只运行单段；
CPU 环境安装与权重准备、真实集成测试命令详见已有接口和验证文档。
已存在的 v01/v02/v03 目录不能作为复现输出目录覆盖。

后续读取应先核对 video_id、frame_index、timestamp_ms；`DepthFrameResult.matches_frame()`
只允许精确当前帧匹配，`load_depth_map()` 校验 status、schema、语义、文件 hash、路径、dtype 和尺寸。
没有实现跨帧插值或将历史图当当前图。

Spatial Grid Fusion v1 可复用原图像素坐标、真实相对深度矩阵、独立质量状态、provenance
与关联索引。未知状态不能填成空闲格，新的聚合和风险关系需另行设计、独立标注与验证。

## 十二、未完成项与阶段终点

尚缺独立深度/遮挡/占用标注、米制标定、相机姿态、跨帧尺度一致性、像素可靠性、
透明反光和细小障碍物可靠检出、多模型并发资源验证、长时稳定性及真实 OOM 恢复证据。
没有报告深度准确率、误报下降或安全收益。

Depth Integration v1 的模型、数据契约、真实八段实验及兼容性验证已经交付。
本阶段到此停止，后续空间融合、走廊策略与反馈需要单独任务授权。

详细资料：

- [模型、接口与复现文档](../research/depth_integration_v1.md)。
- [环境、哈希、实验与验证记录](depth_integration_validation.md)。
- 本地最终实验：`outputs/depth_integration/v03/run_metadata.json`。
- 本地完整校验与回归：`outputs/depth_validation/v02/validation.json`。
- 本地 CPU/CUDA benchmark：`outputs/depth_validation/v01/validation.json`。
- 本地困难案例索引：`outputs/depth_validation/v02/manual_review.json`。
