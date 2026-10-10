# Real-World Benchmark v0.1：首批开发数据

本轮建立真实视频来源、处理、感知重放及独立标注入口。视频只在本机 D 盘保存，
生产风险配置、事件算法、Hybrid 提醒权限和 Phase 3 冻结结果均未修改。
首批是开发诊断数据，不是道路安全验证或独立测试集。

## 来源与许可核查（2026-10-11）

- **PEDESTRIAN**：指定 [Zenodo DOI](https://doi.org/10.5281/zenodo.10907945)
  对应记录 API、指定 [CYENS 仓库](https://github.com/CYENS/PEDESTRIAN)访问返回 404。
  数据/代码许可、文件格式和单文件下载均未确认，不下载，不使用非官方镜像补全。
- **SANPO**：[官方仓库及数据说明](https://github.com/google-research-datasets/sanpo_dataset)
  明确数据为 CC BY 4.0；仓库代码 LICENSE 为 Apache 2.0。真实子集是头部/胸部步行
  PNG 序列，包含相机、姿态、IMU、深度等文件，需排除合成子集。
  官方 GCS 支持单对象读取，但完整数据约 6 TB。抽查首个真实会话的头部左相机，
  元数据为 15 FPS、2208×1242，五张图片每张约 3.66–3.91 MB，按原帧率取得
  八段 10 秒预计超过 4 GiB。本轮只登记来源，未下载视频帧。
  发布者说明人脸/车牌做了模糊处理，仍需当地画质与隐私复核。
- **UrbanEgo**：[官方记录 v1.0](https://zenodo.org/records/23062929)及
  [记录 API](https://zenodo.org/api/records/23062929)明确数据为 CC BY 4.0，
  允许在署名、注明修改的条件下进行本地研究处理。八次 HoloLens 2 头戴步行记录，
  同一拍摄者，葡萄牙 Aveiro，白天晴好天气；含 RGB MP4、深度、IMU/GPS/头部姿态
  及导出检测。未取得独立代码许可，不以数据许可推断代码许可。
  不把发布者检测结果当作风险真值。本轮仅提取 RGB，不根据传感器文件猜测真实路径。
- **EgoBlind**：[官方仓库](https://github.com/doc-doc/EgoBlind)及其
  [LICENSE](https://raw.githubusercontent.com/doc-doc/EgoBlind/main/LICENSE)规定处理数据/
  标注为 CC BY-NC 4.0，面向非商业研究、教育、评价；原始视频可能另有第三方限制，
  受限原文件不可再分发。没有据此推断通用代码许可证。发布者描述 1392 个视障使用者
  第一视角 VideoQA 视频、约 5.3K 问题，有官方 Drive 入口；本轮未下载，后续需逐文件
  审核原始来源权利，适合需求分析。

CC BY 4.0 条款入口：[许可证](https://creativecommons.org/licenses/by/4.0/)。
详细来源字段保存于 `datasets/real_world/manifests/source_catalog.json`，官方响应快照
保存于 Git 忽略的 `reports/source_checks/`。不得将公开仓库等同于视频授权。

## 实际取得的数据

获用户明确批准后，仅下载官方 `run8_20260714_174138.zip`：

- 下载字节：596,342,239（约 569 MiB），小于本轮 1 GiB 总预算。
- 发布者 MD5：`98fc2cd3b807ac3b89816b67c129ee5e`，下载后校验一致。
- 归档 SHA256：`f9e9298672888d414bb490b6df89e09f64f5d1bdf5d1c0535ac013a0ac6b74e7`。
- 仅提取 `run8_20260714_174138/rgb.mp4`，忽略 macOS `._rgb.mp4` 元数据伪文件；
  原 RGB 文件 315,783,434 字节。保留原归档和原视频，不整体解压其他传感器数据。
- 最终首批 **8 段 × 15 秒，共 120 秒**，全部 1280×720、20 FPS、每段 300 帧。
  一次长视频裁剪出的八段，不是八次独立采集。30 段与场景配额尚未完成。
- 原视频 619.1 秒；最终八段研究副本合计 87,462,566 字节，另有被替换候选。
- 开始时刻（原 MP4 秒）：20、75、115、226、302、377、453、490；区间不重叠。
  所有片段同一 parent_source_id，全部 development。随机种子 20261011，暂无 test。
- attribution：Rodrigo Abreu、André Clérigo、Gonçalo Lourenço Silva、Pedro Rito、
  Susana Sargento；UrbanEgo v1.0；DOI 10.5281/zenodo.23062929。派生数据注明裁剪和重编码。

位置：`D:\program\SightGuard\datasets\real_world\raw\urbanego\`（原始），
`processed\`（裁剪）；模型输出使用全新 `outputs\real_world\v01\`。
归档下载量不等于本地占用量，本地另包含原视频、候选/替换片段和标注视频。
人脸、车牌可能可识别，所有视频、接触表和关键帧保持本地，未上传/提交公共仓库。

## 标准化与画质

使用已安装 imageio-ffmpeg 的 FFmpeg，H.264 CRF 18，无音轨，不缩放、不重新采样 FPS。
OpenCV 完整解码原视频和各片段；检查帧数、严格递增 PTS、文件 SHA256、解码像素指纹、
重复帧比例和低分辨率帧间差异。最后八段没有相同文件哈希或相同解码像素指纹。
这些只能排除明显重复，不能证明不同片段在场景上独立。

每段保存源与输出逐帧时间映射，帧数一致，相对时间偏差不超过 1 ms。
这是 **MP4 presentation time** 映射，不是 HoloLens 主机时钟、IMU/GPS 同步证明。
不可据编码 FPS 虚构传感器采样时间。

质量复核采用每段 1、7.5、14 秒抽帧接触表，记录为 agent quality review，
不是全程人工观看，更不是人工风险真值。初始开头设备调试、明显画面异常、
末尾墙面静止候选被替换，候选仍在本地保留；见 `reports/rejected_candidates.json`。
街道、停放车辆、行人、店铺、桌椅等画面可以观察到；静态障碍/动态目标/无风险配额
尚未人工确定，统一场景标签 `urban_pedestrian_walk_unlabelled`。
后续人工标注必须检查完整片段中的模糊、转向、遮挡与无法判断区间。

## 数据格式与复现

来源 manifest 使用 `sightguard.real-world.v0.1`，定义在 `manifest.schema.json`；
视频 ID、父来源、许可、获取/审核状态、尺寸/FPS、原/处理哈希、裁剪起止和处理记录
保存在 `manifests/dataset.json`。它只管理来源，不替代 Phase 3 Ground Truth Schema 3.0。
Python 校验器实施接受状态、文件 SHA256、重复与分组约束，不声称执行完整 JSON Schema 校验。

Windows 每次新 PowerShell 运行均先配置 D 盘缓存：

```powershell
Set-Location D:\program\SightGuard
$env:TEMP = "$PWD/models/hybrid/tmp"
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:UV_CACHE_DIR = "$PWD/models/hybrid/cache/uv"
$env:HF_HOME = "$PWD/models/hybrid/cache/hf"
$env:TORCH_HOME = "$PWD/models/hybrid/cache/torch"
$env:YOLO_CONFIG_DIR = "$PWD/models/hybrid/cache/ultralytics"
$env:PYTHONDONTWRITEBYTECODE = '1'
```

下载函数 `evaluation.real_world.download_object` 要求 HTTPS、明确大小和发布者 MD5；
大于 500 MiB 需明确批准，失败只留下 `.part` 与 failed 记录，不产生 verified 文件。
本轮归档已经下载，不重复获取。`extract_member` 只提取安全的指定文件，不覆盖已有对象。

在**空白首批目录**重建最终选择（既有 manifest/原视频/输出不会覆盖）：

```powershell
uv run python scripts/prepare_real_world.py --archive datasets/real_world/raw/urbanego/run8_20260714_174138.zip --starts 20 75 115 226 302 377 453 490
```

准备脚本保持 `pending_visual_review`；确认画质后手动更新质量审核与 accepted，不能把它
更新为人工风险 reviewed。随后调用已有管线：

```powershell
uv run python scripts/run_real_world.py
# 仅恢复核验一致的已完成感知/评估，不覆盖异常或不同配置产物：
uv run python scripts/run_real_world.py --resume
```

研究感知配置 `configs/real-world-perception.yaml` 为本地 YOLO26n、CPU、640、阈值 0.25、
BoT-SORT、全部 COCO 类别；扩大类别仅用于本数据研究，生产默认配置未改。
风险配置继续使用 `configs/risk.yaml`；A/B/C 协议继续使用 `configs/benchmark.yaml`。

## 独立标注与评价流程

Phase 3 导入器创建 `annotations/phase3/manifest.csv`、逐视频 Schema 3.0 draft JSON，
感知 JSONL 和配套 Phase 1 元数据复制到该目录的 tracks 子目录，保留真实来源证据。
模板 events 为空不代表负样本。风险日志/模型框不得提供给独立标注者。

现有 Research Console 数据集页是**只读**，有预览、数据检查和无 GT 策略预览；
没有事件区间可视化编辑器。本轮采用已有 JSON + CLI 标注/校验流程，未新增管理网页。
启动 `uv run sightguard ui --research --port 7861`；数据集选择
`real_world/annotations/phase3/manifest.csv`。原片可用本地播放器查看，编辑对应 JSON，
遵循 [标注规范](annotation_guidelines.md)，填写事件区间、首次/升级机会、人工目标区域
和排除区间。无风险片也须实际审核，记录 annotator_id、human_raw_video 来源、复核分歧。

视觉属性独立任务清单在 `annotations/visual_tasks.json`，关键帧保存在本地
`annotations/visual_frames/`，含图像哈希和视频时间，标签默认 null；涵盖可见性、
左/中/右、候选可见通行区域重叠、遮挡、画面运动及证据充分性，不推断碰撞。
这些是待人工填写的 sidecar，不进入 Phase 3 风险真值，也不自动转换为 Laya 标签。

审核完成后按既有入口执行（本轮没有执行成功的监督 benchmark）：

```powershell
uv run sightguard seal-labels --dataset datasets/real_world/annotations/phase3/manifest.csv
uv run sightguard validate-labels --dataset datasets/real_world/annotations/phase3/manifest.csv
uv run sightguard benchmark --dataset datasets/real_world/annotations/phase3/manifest.csv --split development --config configs/benchmark.yaml --risk-config configs/risk.yaml --output-dir outputs/real_world/reviewed_benchmark_v01
```

只修改已实际人工审核的文件；部分审核时建立仅含 reviewed 样本的新 manifest，不能批量
改状态。本轮 **人工风险标注 0，人工视觉属性标注 0**。监督指标闸门必须拒绝 draft，
没有 TP/FP/FN、召回率、误报率、视觉正确率。算法提醒计数仅是行为日志。
Stage 3 按任务要求停在待标注状态；本轮未执行 Laya Vision 自然场景推理或准确率实验，
UNKNOWN 不解释为风险等级，Hybrid 继续 Shadow Mode。

后续失败案例按 YOLO 漏检/类别错误、ID 切换、评分不足、无风险误报、过晚/重复提醒、
视觉语义误判、UNKNOWN、输入不足分类。必须将人工 GT 与输出在同一时间段对照后登记，
保留视频 ID、配置、区间、标签和假设，不从未标注的提醒日志宣称真实失败因果。

## 当前验证与限制

新增测试覆盖失败下载、校验、不覆盖、视频元数据与裁剪 PTS、分组泄漏、重复/篡改、
Schema 3.0 草稿拒绝及 UNKNOWN。新增 10 项测试通过；全量回归命令使用本地模型与
Hybrid CPU/GPU 验证开关（结果见本节最后的实际记录），未为测试下载模型。

真实批处理完成八段，每段 300 感知帧、300 风险帧，共 11,293 条目标检测、
13,852 条事件变化记录、257 次规则提醒。事件变化记录数量不等于独立事件数量。
这些是未经 GT 判断的算法输出，不是 TP、误报或重复提醒指标。
逐视频运行保存模型/配置/输入哈希、依赖版本和 PTS 来源，可独立重放。

ConsoleStore 数据集浏览成功读取八条真实记录，全部分析视图均读取 300 感知帧与
300 风险帧，无遗漏警告。既有 A/B/C 策略预览成功使用八份相同感知数据，
返回 metrics={}、ground_truth_evaluated=false。没有生成正式对照指标。
真实 `sightguard benchmark --data-kind real --split development ...` 调用在
dataset_validation 阶段按预期拒绝 draft，返回退出码 1，未生成 metrics.json。
见本地 `outputs/real_world/unreviewed_gate_check/run_metadata.json`。

实际执行：准备脚本、合成视频测试、真实批处理（导入元数据修复后 --resume）、
控制台服务接口、CLI 草稿拒绝、`ruff check .`、`ruff format --check .` 和 pytest。
初次导入遗漏配套 Phase 1 元数据时被现有校验器阻止；修复仅在新脚本中保留来源证据，
未弱化 Phase 3 校验。浏览器页面交互未在本轮单独验收。

未解决：独立拍摄来源不足、30 段配额未达、完整人工画质复核与独立 GT 尚缺；
没有正式真实风险性能结论，也没有 Laya 自然场景准确性结论。

最终检查：`uv run ruff check .`、`uv run ruff format --check .` 通过。
`uv run pytest -q --basetemp models/hybrid/tmp/pytest-real-world-final`：
**332 passed，67.22 秒，无跳过**。运行时设置 `SIGHTGUARD_TEST_MODEL=models/yolo26n.pt`
（绝对 D 盘路径）、`SIGHTGUARD_TEST_HYBRID=1`、`SIGHTGUARD_TEST_HYBRID_GPU=1`，
覆盖现有本地真实模型集成测试。新增数据工具测试独立运行 **10 passed**。
