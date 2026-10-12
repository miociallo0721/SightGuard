> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard 阶段性报告：Hybrid Intelligence 与 GPU 可行性研究

报告日期：2026 年 10 月 11 日。项目目录：`D:\program\SightGuard`。
本报告汇总既有 Hybrid Intelligence 原型及 10 月 10 日 GPU 研究的交付状态；本次编写报告未重新运行模型或性能测试。

## 1. 当前阶段结论

SightGuard 已具备本地视觉感知、动态风险事件、离线对照实验、实时摄像头、网页研究界面及空间音频的工程基础。在此基础上，Laya Structured 和 Laya Vision 已完成真实 CPU 推理并接入实时影子研究通道。

现阶段实际提醒与空间音频仍由原规则引擎生成。辅助模型输出用于研究记录、同源结果对照和实验性标签融合，不直接改变 AlertEvent。项目尚未证明多模型融合能提升重要风险检出或减少无效提醒，也未形成可靠碰撞预测或安全保证。

最近一轮交付完成了统一 Benchmark、资源竞争测试工具、CPU 复核及运行时失败诊断。**尚未完成 GPU 模型推理验证**，不能宣布 RTX 2050 上的 FP16 或 ONNX CUDA 加速成功。

## 2. 已完成的工程工作

### Hybrid Intelligence 原型

- 使用真实 Laya Structured 与 Laya Vision 权重，固定模型及代码版本；各用隔离环境和独立模型进程，避免同名 laya 包冲突。
- 复用既有 YOLO26n、BoT-SORT、Phase 2 风险评分、状态机及空间音频，不重新实现这些算法。
- 建立结构化语义输入、原始输出解析、模型加载/推理/卸载、超时处理及模式切换隔离。
- 旁路语义线程使用最新输入槽，不要求快速规则链等待辅助模型；默认 1 Hz、2 秒结果有效期。
- 原网页提供 Rule、Laya Structured、Laya Vision、Hybrid 研究模式及模型状态、结果对照和耗时信息。
- 过期、UNKNOWN 或无法匹配当前帧/事件的候选不能参与当前有效比较。UNKNOWN 表示弃权，不是风险候选。

上述是既有阶段交付，详细来源、许可证、历史摄像头/网页验证及限制见 [Hybrid Intelligence 交付报告](hybrid_intelligence_delivery_report.md)。

### 本轮 GPU 研究与必要兼容修复

- 新增独立环境准备脚本，新增环境、缓存、下载与临时数据放在 D 盘，不覆盖原 CPU 环境。
- 新增固定输入 Benchmark，保存权重/输入/代码哈希、软件版本、加载与冷启动耗时、30 次稳态记录及原始输出。
- 分开统计预处理、模型调用和结果解析；GPU 路径包含同步计时与显存记录，Structured 包含 ORT 节点 profiling。
- 新增同源视频 Rule-only / Rule+Structured / Rule+Vision 资源竞争工具，保持原规则链与 2000 ms TTL；该三组 GPU 实验尚未执行。
- 新增 Vision 混合精度适配：计划将 CUDA encoder 设为 FP16，保留上游要求的 FP32 readout/head。记录真实参数 dtype/device；该适配未在 GPU 上实测。
- 针对 Windows CPU 回归中的 safetensors mmap 原生读取故障，隔离进程采用 pread；不修改权重或上游模型代码，读取后恢复原函数。
- 权重哈希验证移至模型加载之前；修复 Benchmark 将合法 UNKNOWN 文本误判为非法结果的问题，并保留旧失败记录。

AGENTS.md、默认 CPU 配置、Phase 2 风险算法、AlertEvent 条件、音频调度及 Phase 3 冻结实验均未修改。

## 3. 实际测量结果

### 硬件与运行环境

10 月 10 日观测到 NVIDIA GeForce RTX 2050 Laptop，4096 MiB 显存，驱动 616.56；nvidia-smi 显示驱动支持 CUDA 13.4。主环境为 Python 3.12.15、PyTorch 2.14.1+cpu，CUDA 不可用。

“系统识别 GPU”“存在 CUDA provider 名称”“模型实际执行 GPU 节点”是不同验证层级，本轮只取得前两层的部分证据。

### 固定输入 CPU 复核

两种模型各预热 3 次，执行 30 次稳态真实推理，均 30/30 成功，零推理异常和非法结果。

- **Laya Structured，CPU INT8**：predict P50 662.923 ms，P95 699.108 ms，平均 665.815 ms；加载 6099.176 ms，冷启动端到端 720.001 ms。节点 profiling 只观察到 CPUExecutionProvider，30 次为合法非 UNKNOWN 输出。
- **Laya Vision，CPU FP32**：predict P50 1701.306 ms，P95 1756.628 ms，平均 1706.765 ms；加载 6726.317 ms，冷启动端到端 1658.897 ms。201,161,347 个参数均在 CPU、FP32，30 次均为合法 UNKNOWN。

这两项使用同一 Benchmark 的固定输入定义，但按各自接口适配任务；它们是调用性能测量，不是两个模型准确率或效果优劣的对照。Vision 虽在这组输入上耗时低于 2 秒，但全部弃权，不能称为产生了有效风险候选。

原始结果位于 `outputs/hybrid/gpu-benchmark/laya-cpu-verified/` 和 `vision-cpu-verified/`。历史网页会话中 Vision P50 约 3829 ms，输入与负载不同，不能与本轮合成输入混用或宣称实时延迟已经改善。

### GPU 初始化结果与阻碍

新 ORT GPU 环境列出 CUDAExecutionProvider，但实际 QInt8 Session 初始化后只有 CPUExecutionProvider。初始化日志指出缺少 `cublasLt64_13.dll`，要求 CUDA 13 与 cuDNN 9；此次只验证了 Session 初始化，没有进行该 Session 的节点 profiling 或推理。

CUDA PyTorch 安装包下载未完成。根据用户在额度耗尽前交付的要求，本轮停止大下载并提交可复用工具与真实记录。准备脚本已改用 cu130，但完整安装尚未跑通。

该结果证明本轮 CUDA 运行时前置条件没有满足，不能证明 RTX 2050 硬件不兼容，也不能证明 QInt8 算子不支持 CUDA。GPU P50/P95、峰值显存、CPU/GPU 输出差异、重载稳定性和真实 OOM 均未测。

## 4. 测试与交付记录

- 完整 pytest 回归：**319 passed，1 skipped，42.85 s**。启用了本地真实 YOLO 与 CPU 辅助模型集成验证；跳过项为 GPU 真模型 opt-in 测试，原因是环境未完成。
- 最终新增测试复核：**9 passed，1 skipped，3.10 s**，覆盖精度适配、UNKNOWN、profiling 解析、计时与结果新鲜度。
- `uv run ruff check .` 通过；`uv run ruff format --check .` 报告 137 个文件已格式化。
- PowerShell 环境脚本语法检查通过；资源竞争 CLI `--help` 可启动。这不等于完成安装或实际三组实验。
- 回归期间临时停止原服务释放内存；10 月 10 日交付前恢复 7860、7861、7862，均返回 HTTP 200。本报告未重新检查当前服务在线状态。

本轮中途发生过内存不足与 Windows mmap 原生故障。兼容修复后完整回归通过；未删除、跳过或弱化原有效测试，未据短测试宣称长期稳定。

业务与研究工具提交：`cf18022`，`feat: add hybrid GPU feasibility study tools`。
分支：`codex/hybrid-gpu-feasibility`。**仅本地提交，未推送远端。**

主要交付文件：

- `scripts/benchmark_hybrid_gpu.py`：独立模型 Benchmark。
- `scripts/benchmark_hybrid_pipeline.py`：同源视频资源竞争工具。
- `scripts/prepare_hybrid_gpu_environment.ps1`：D 盘隔离环境准备。
- `src/sightguard/hybrid/runtime_support.py`、`model_host.py`：读取与精度适配、真实设备记录。
- `configs/hybrid-gpu-study.yaml`：独立研究配置，未替换默认配置。
- 三个新增测试文件，以及 [GPU 可行性研究记录](../research/hybrid_gpu_feasibility.md)。

模型、环境、私人视频、下载片段、运行产物均位于 Git 忽略目录，不随提交分发。此前的 `hybrid_intelligence_delivery_report.md` 尚未纳入本轮业务提交，编写本报告未覆盖它。

## 5. 下一阶段的优先顺序

1. 完成独立 CUDA 运行时安装，先核验 torch 实际 CUDA 构建与可用性、ORT 实际 Session providers；避免仅检查 provider 名称。
2. 优先跑真实 Laya Vision FP16/必要 FP32 混合推理，确认参数位置、输出合法性，再完成相同输入 CPU/GPU 对照及各 30 次稳态测量。
3. 对 Structured QInt8 跑真实 CUDA profiling，区分 CPU fallback 与 GPU 节点；仅在确有收益时考虑保留 GPU 路径。
4. 保持 2 秒 TTL，执行同一视频三组资源竞争，测量规则 FPS、候选完成年龄、过期与音频 worker 状态。软件状态不能代替耳机试听。
5. 工程可行性通过后再建立独立标注的风险效果对照，评估重要风险检出、无效与重复提醒，不能用模型成功调用代替研究有效性。

当前适合继续开发和可复现实验准备；**尚不适合宣称 GPU 加速验收完成、多模型风险效果已证实或真实道路辅助安全性已验证。**
