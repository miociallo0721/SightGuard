> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Phase 1 实际验证记录

验证日期：2026-10-09（Asia/Shanghai）。以下是本次在 Windows 工作区实际运行的结果，
不是性能基准、风险识别准确率或使用者试验。

## 环境

- Windows x86_64；系统 Python 3.13 不用于项目，uv 独立安装 Python 3.12.15。
- SightGuard 0.1.0；Ultralytics 8.4.174；torch 2.14.1；torchvision 0.29.1。
- NumPy 2.5.3；opencv-python 4.14.0.94；lap 0.5.13；完整解析见 uv.lock。
- 本次 PyTorch 环境 `cuda_available=false`、`torch_cuda=null`，推理设备为 CPU。
- 官方权重：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`，
  保存到 models/yolo26n.pt，不提交 Git。
- 权重 SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

## 实际命令与结果

按 README 在项目根目录运行；下载模型仅在准备阶段进行。

```powershell
uv python install 3.12
uv sync
uv sync --locked
uv run sightguard --help
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

Python 3.12 安装成功，locked sync 成功。初次 sync 在 README 尚未创建时构建失败，
补齐 README 后重试成功。uv 报跨盘 hardlink 回退为 copy，未影响安装结果。
开发中的 Ruff 行宽/导入顺序问题经格式化修正，最后 `ruff check` 和
`ruff format --check` 均通过。CLI help 能启动，wheel 与 sdist 构建成功。
最终默认 pytest：**24 passed, 2 skipped**（1.01 秒）；跳过的两项是显式 opt-in 模型测试，
默认未指定模型，不是忽略已知故障。该耗时仅是此次测试报告。

```powershell
uv run python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt')"
uv run python scripts/generate_demo.py
uv run sightguard analyze --input examples/demo.mp4 --output-dir outputs/demo --config configs/default.yaml
uv run python scripts/generate_tracking_fixture.py
uv run sightguard analyze --input examples/tracking_fixture.mp4 --output-dir outputs/tracking_verified --config configs/default.yaml
uv run sightguard replay --input outputs/tracking_verified/tracks.jsonl
$env:SIGHTGUARD_TEST_MODEL = (Resolve-Path models/yolo26n.pt).Path
uv run pytest -q -m integration
uv run pytest -q
Remove-Item Env:SIGHTGUARD_TEST_MODEL
```

真实 CPU 集成测试：**2 passed, 24 deselected**（4.39 秒）。随后带本地权重运行完整套件：
**26 passed**（5.31 秒），无跳过。模型测试包含合成无目标视频与重复真实图像视频。
准备权重时成功从官方 GitHub 下载；默认单元测试及模型测试本身均不联网下载权重。
命令的生成与输出目录均拒绝覆盖已有文件，重跑时请选新输出路径。

合成视频分析成功生成三种产物：20 帧、空检测结果。BoT-SORT 相机运动补偿在无纹理
矩形画面上打印 `not enough matching points`，本次没有抛异常，所有帧均保留。

真实目标冒烟视频来自已安装 Ultralytics 的 assets/bus.jpg，静态图像缩放后重复：
10 帧、10 FPS、480×640。分析和 replay 成功，40 条检测记录、0 个失败帧；
每帧 4 条检测，追踪 ID 1、2、3、4 跨全部 10 帧出现。
对标注 MP4 进行完整逐帧解码检查：10 帧、尺寸一致、可解码。
另有 outputs/tracking_smoke 保留早期验证产物，最终代码运行产物是 outputs/tracking_verified。
原图与视频哈希存于 examples/tracking_fixture.provenance.json；运行元数据记录模型与跟踪 YAML 哈希。

## 测试范围与限制

单元测试覆盖：配置有效/无效加载、JSONL 序列化和回放、schema/顺序验证、视频 PTS 与 FPS 回退、
无目标帧、缺失 ID、模拟结果转换、跟踪参数传递、每视频新模型会话、异常输入、首帧/中途读取失败、
空解码帧、感知异常失败帧保存、编码器失败、输出视频逐帧解码与拒绝覆盖。

GPU、Linux/macOS、真实动态街景、低光/遮挡/密集场景、可变帧率精确时间恢复、
实际跟踪准确率与提醒效果均未验证。静态图像重复验证只能证明链路和 ID 关联接口，
不能作为运动跟踪性能数据。没有收集带独立标注的研究数据集，也没有风险算法或提醒策略结果。
OpenCV 总帧数缺失时仍无法可靠区分 EOF 与解码失败，运行元数据会记录这种不确定性。

## Phase 2 开始前的技术决策

- 明确首批目标类别、数据采集场景，以及相机佩戴位置和画面区域假设。
- 确定仅基于二维框的可解释风险特征及措辞边界，避免把相对框变化解释为已验证碰撞预测。
- 明确跟踪 ID 丢失/切换时事件关联与终止策略，以及失败/时间回退帧对事件状态机的影响。
- 确认风险确认时长、提醒冷却、升级条件和提醒渠道的配置与测试方案。
- 规划独立事件标注与数据划分，之后以固定感知 JSONL 比较各策略。
