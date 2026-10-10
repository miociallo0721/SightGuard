# Hybrid GPU 可行性研究记录

研究日期：2026-10-10。工作分支：`codex/hybrid-gpu-feasibility`。本轮提交研究工具、CPU 复核和真实失败诊断，不代表 GPU 验证通过。

## 实际结论与边界

系统识别 RTX 2050 Laptop，但尚未获得任何辅助模型的 CUDA 推理结果。CUDA PyTorch 大型安装包下载未完成；用户要求在额度耗尽前交付，因此停止下载。新隔离环境的 ORT GPU 初始化实际回退 CPU：其 Windows 1.31.0 包要求 CUDA 13 DLL，当前缺少 `cublasLt64_13.dll`，并报告需要 CUDA 13 与 cuDNN 9。这是运行时依赖问题，不能据此断言 GPU 硬件不兼容或模型 INT8 算子不支持。

默认 CPU 配置、Phase 2 风险评分、AlertEvent 条件、音频调度、冻结实验及 AGENTS.md 均未修改。建议继续使用现有规则链和 CPU 影子模式；GPU 配置仅供研究，不作为实际提醒依据。

## P0：环境观测

- GPU：NVIDIA GeForce RTX 2050，4096 MiB，WDDM；初始可用显存 4096 MiB。
- NVIDIA 驱动：616.56；`nvidia-smi` 报告驱动支持 CUDA 13.4。这不等于已安装 CUDA PyTorch。
- 主环境：Python 3.12.15，PyTorch 2.14.1+cpu，`torch.version.cuda=None`，`torch.cuda.is_available()=False`。
- 原 Structured CPU 环境：ORT 1.31.0，Azure/CPU providers。
- 新 Structured GPU 环境：可用 provider 列表含 CUDA/TensorRT/CPU，但实际 QInt8 Session 只有 CPUExecutionProvider，初始化约 1979.675 ms。此次仅验证初始化，没有执行该 Session 推理或节点 profiling。

原始记录：`outputs/hybrid/gpu-benchmark/environment-initial.json` 和 `ort-cuda-probe/probe.json`。原生错误出现在初始化日志；available providers 与实际 Session providers 必须区分。

## P1–P2：必要适配与未验证项

固定版本 Vision 的 `_readout` 显式将隐藏状态转换成 FP32。新增适配保持 readout/head FP32，仅 CUDA encoder 使用 FP16，记录实际参数 dtype/device 数量。这是 `fp16_encoder_fp32_readout`，不是全模型 FP16。本轮未在 GPU 上执行全 FP16 或混合精度推理，未验证 CUDA attention、输出差异、加载/冷启动、峰值显存、重载或真实 OOM。

CPU 回归发现 Windows safetensors 默认 mmap 读取有原生访问冲突。隔离模型进程中使用官方 `load_file(..., backend="pread")`，调用完成恢复原函数；不修改上游代码或权重。真实 CPU 模型加载及完整回归通过，不能据此推断 GPU 稳定性。[safetensors 官方读取接口](https://huggingface.co/docs/safetensors/api/torch)

Structured 保留既有 QInt8 权重。CUDA provider 初始化失败发生在节点分配验证之前，因此 INT8 节点分配、CPU fallback 比例、CUDA 延迟及可选 FP32 CUDA 均未测。

## P3：统一测试工具与 CPU 复核

`scripts/benchmark_hybrid_gpu.py` 使用固定合成图像与语义输入，记录输入/权重/源码哈希、版本、模型加载、冷启动、预热后 30 次推理、原始判断、预处理/模型/解析分项、CUDA 同步、显存采样和 ORT 节点 profiling。新输出目录不得覆盖已有结果。

早期 `vision-cpu-original` 完成 30 次真实推理，模型 predict P50 1724.340 ms、P95 1777.276 ms、平均 1721.310 ms，输出为合法 UNKNOWN。该旧记录的脚本错误地把 UNKNOWN 的自然语言 choice 判为 invalid_output，已修复并增加测试；原始记录保持不变。早期 Structured 两次启动失败分别为缺 Pillow、runpy 启动未加入 scripts 路径；不计为成功推理。

最终复核两种模型均 success，各预热 3 次、稳态 30/30 成功，零推理异常与非法结果：

- Structured CPU INT8：predict P50 **662.923 ms**、P95 **699.108 ms**、平均 **665.815 ms**；加载 6099.176 ms，冷启动端到端 720.001 ms。ORT profiling 实际只有 CPU 节点，30 次为合法非 UNKNOWN 输出。
- Vision CPU FP32：predict P50 **1701.306 ms**、P95 **1756.628 ms**、平均 **1706.765 ms**；加载 6726.317 ms，冷启动端到端 1658.897 ms。201161347 个参数均 CPU/FP32，30 次均为合法 UNKNOWN，不能当作有效风险候选。

记录位于 `laya-cpu-verified/` 和 `vision-cpu-verified/`。Structured 用原 CPU 解释器，以只读 sys.path 追加新 Vision 环境的 Pillow，未安装或覆盖原 CPU 环境；runpy 明确加入 scripts 路径。Vision 直接运行原 CPU 解释器。两次均使用同一脚本固定输入定义，模型任务仍按各自接口适配。CPU 数值不能推导 GPU 提速、显存或真实会话 2 秒 TTL 下的有效候选比例；固定输入性能也不能代表网页负载。

## P4：资源竞争工具

`scripts/benchmark_hybrid_pipeline.py` 使用同一 MP4，分别创建 Rule-only、Rule+Structured、Rule+Vision 会话；保留最新帧覆盖机制和 2000 ms TTL，统计实际规则更新、完整帧处理、候选完成年龄、过期、错误与设备显存。可接入现有空间音频 worker，软件状态不等于人耳验收。

GPU 前置条件未满足，本轮没有执行三组 GPU 竞争实验，没有真实 GPU FPS、TTL 命中率、音频竞争或长期稳定性结论。工具的 CPU 模拟测试已验证年龄/UNKNOWN/风险计时等统计逻辑。

## 复现与后续验证

所有新增环境、缓存、临时文件与输出在 D 盘；原 CPU 环境保留。准备脚本不下载模型，沿用已有固定模型与上游 revision。PyTorch 下载先尝试 cu128，发现 ORT CUDA 13 要求后将脚本改为官方 cu130；修正后的完整安装流程尚未跑通，不得视为环境已准备成功。[PyTorch 官方安装源](https://download.pytorch.org/whl/cu130/torch/)、[ORT CUDA 依赖说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

在项目根目录 PowerShell 执行（准备脚本有较大下载，需另行安排）：

```powershell
./scripts/prepare_hybrid_gpu_environment.ps1
# 必须先确认两个隔离环境 CUDA 可用及 Session 实际 providers。
./models/hybrid/env-vision-gpu/Scripts/python.exe scripts/benchmark_hybrid_gpu.py --model vision --device cuda --precision fp16 --iterations 30 --output outputs/hybrid/gpu-benchmark/vision-cuda-01
./models/hybrid/env-structured-gpu/Scripts/python.exe scripts/benchmark_hybrid_gpu.py --model laya --device cuda --precision int8 --iterations 30 --output outputs/hybrid/gpu-benchmark/laya-cuda-01
uv run python scripts/benchmark_hybrid_pipeline.py --help
```

本轮 CPU 复核命令（输出目录应换为新的未存在目录）：

```powershell
$env:TEMP="$PWD/models/hybrid/tmp"; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
$env:UV_CACHE_DIR="$PWD/models/hybrid/cache/uv"
./models/hybrid/env-structured/Scripts/python.exe -c "import sys,runpy; from pathlib import Path; sys.path.insert(0,str(Path('scripts').resolve())); sys.path.append(str(Path('models/hybrid/env-vision-gpu/Lib/site-packages').resolve())); sys.argv=['scripts/benchmark_hybrid_gpu.py','--model','laya','--device','cpu','--precision','int8','--output','outputs/hybrid/gpu-benchmark/laya-cpu-verified','--iterations','30','--warmup','3']; runpy.run_path('scripts/benchmark_hybrid_gpu.py',run_name='__main__')"
./models/hybrid/env-vision/Scripts/python.exe scripts/benchmark_hybrid_gpu.py --model vision --device cpu --precision fp32 --output outputs/hybrid/gpu-benchmark/vision-cpu-verified --iterations 30 --warmup 3
```

GPU 模型测试后再运行同源视频竞争工具，配置 `configs/hybrid-gpu-study.yaml`，保持 TTL。先检查 benchmark 原始结果/实际设备/节点 provider，再判断是否有加速。不要在运行时自动改变默认 CPU 配置。

## 自动化检查

完整 pytest 回归实际结果：**319 passed，1 skipped，42.85 s**。显式开启现有真实 YOLO 和 CPU 辅助模型集成测试；跳过项为 opt-in GPU 真模型测试，原因是 CUDA 环境未完成。新增测试覆盖精度适配、UNKNOWN 合法性、ORT profiling、计时边界、结果新鲜度和风险计时。测试不替代真实 GPU 推理。

实际检查命令：`uv run ruff check .`（通过）、`uv run ruff format --check .`（137 文件已格式化），以及 `uv run pytest -q --basetemp models/hybrid/tmp/pytest-gpu-reader-full`。完整回归前设置 `SIGHTGUARD_TEST_MODEL` 为本地 `models/yolo26n.pt` 绝对路径、`SIGHTGUARD_TEST_HYBRID=1`，TEMP/TMP/TMPDIR 和 UV_CACHE_DIR 如上述 D 盘配置；GPU opt-in 未启用。PowerShell 环境准备脚本另经 Parser 语法检查通过。

本轮中途因服务占用内存及 Windows mmap 原生故障发生过失败；隔离 CPU 小模型测试避免主测试进程持有 Torch，使用 pread 后完整回归通过。未删除、弱化或跳过原有效测试。服务仅临时停止以释放内存，交付前恢复原入口。Git 仅提交源码/配置/测试/文档，不提交环境、模型、私有视频、下载片段或实验运行产物，不推送远端。
