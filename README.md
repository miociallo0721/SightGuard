# SightGuard（视界）

面向视障辅助的离线环境感知应用。使用摄像头识别画面中的目标和环境结构，在证据满足条件时给出简短中文提醒，并抑制持续事件的重复播报。

**当前阶段：Android 0.4 功能预览，附带独立 PC 评测工具与默认关闭的研究模块。尚未达到公开产品发布标准。** SightGuard 补充盲杖和导盲犬；没有提醒不代表前方安全，环境候选区域也不代表人体可以通过。

## 可以做什么

- 前台打开后自动开始感知；支持主动停止、静音、恢复提醒和本地视频分析。
- 使用 YOLO26n 检测目标，通过连续观测、规则评分和事件调度筛选需要提醒的对象。风险分数是图像证据的启发式评分，不是碰撞概率。
- 自动观察墙面、楼梯、杆柱和栏杆，在确认新事件后进入统一中文语音链路。普通用户不需要选择模型或打开 Mask。
- 简洁预览、当前有效提醒和最近一次实际播报分别显示；研究图层和诊断位于“设置 → 开发者工具”。
- 固定提醒有内置中文离线音频；主动“描述周围”需要可用的离线动态中文 TTS，否则显示描述文本和无法朗读的原因。

单级台阶和路缘尚未可靠识别，自动台阶提醒关闭。楼梯提示不区分上行、下行或地面落差。门、树、围栏等用于环境概览，不自动推断出口、近处障碍或通行安全。

## 当前不提供的能力

没有经过校准的墙距播报、身体接近判断、真实通道宽度、安全路径规划或自动绕障。米制深度、墙面接近和局部空间候选只用于研究，默认不改变既有风险判断和声音输出。

已发现夜间楼梯误识别为墙面、细杆及台阶漏检、局部候选落在海报平面等失败。详见[当前能力与验证范围](docs/product/status.md)和[尚未解决的问题](docs/product/known-issues.md)。

## Android 使用

需要 Android 8.0 / API 26 以上的兼容设备。APK 不申请 INTERNET 权限；模型和固定音频准备好后在本地运行。首次使用需要正常授予摄像头权限。

1. 安装经过核验的 APK，打开应用并授予摄像头权限。
2. 查看当前状态；应用在前台自动感知，主动停止后不会自行重新开始。
3. 使用“描述周围”“静音 / 恢复提醒”“停止 / 开始感知”“设置”四个主要操作。

**最近完成较完整真机回归的是 `0.4-accessible-awareness`。** 本分支的 `0.4-metric-corridor-shadow` 已构建并在后续任务中安装、确认前台打开，但新增米制模块尚未完成 Android 推理和性能验收。不能用前一版本的验收替代这一项。

仓库不包含 APK 或大模型，也没有承诺可下载的正式发行包。本机保留的稳定 / 实验包、SHA256 和回退方式见[版本与交付](docs/product/releases.md)。源码构建、模型准备和 ADB 测试见 [Android 开发说明](android/README.md)。

## PC 运行与评测

PC 是本地演示与评测工具，不等同 Android 的模型参数、跟踪器和语音实现。需要 Python 3.12 和 uv：

```powershell
uv sync --locked
uv run sightguard --help
uv run sightguard ui --port 7860 --open-browser
```

浏览器打开 `http://127.0.0.1:7860`。界面启动不下载模型；执行视频或摄像头识别前须准备本地 YOLO 权重。原研究控制台单独使用 `uv run sightguard ui --research --port 7861`。

如果只想验证核心事件链路，可以直接重放仓库中的合成输入，不需要模型或 GPU：

```powershell
uv run sightguard assess --tracks tests/fixtures/phase2/static_occlusion.jsonl --config configs/risk.yaml --stream-id synthetic-01 --output-dir outputs/first-run/risk
```

输出目录必须是新的目录。该命令生成结构化风险、事件和提醒日志，不播放语音，也不证明真实目标识别效果。模型准备、真实视频分析和摄像头命令见 [PC 使用说明](docs/pc-guide.md)。

## 项目结构

```text
android/                 Android 应用、纯 Kotlin 核心、设备测试与工具
src/sightguard/
  perception/            目标检测与跟踪适配
  pipeline/              视频分析与感知结果重放
  risk/ decision/        规则评分、事件生命周期与提醒筛选
  live/ spatial/         PC 实时适配与可选本地空间音频
  scene/ research/       图像空间证据及独立研究模块
  evaluation/            独立标注、协议冻结与对照评测
  ui/                    PC 产品演示页与研究控制台
configs/                 版本化配置
tests/                   离线自动化测试与小型夹具
scripts/                 显式运行的数据、模型和评测工具
docs/                    使用、架构、问题清单及历史验证记录
```

模型、原始数据、APK 和运行输出保留在 Git 忽略的本地目录，不随源码发布。[文档索引](docs/README.md)区分当前说明和历史实验记录；[架构说明](docs/architecture.md)区分正式代码链路与 Shadow 研究。

## 开发与贡献

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

默认测试不下载模型、不要求 GPU。真实模型、Android instrumentation 和真实使用者验证是独立证据等级，不可相互替代。提交要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，隐私与问题报告见 [SECURITY.md](SECURITY.md)。

当前尚未确定仓库整体的公开许可证，也未完成全部权重、数据及依赖的发行许可审查。不要将公开可读源码解释为已经获得任意用途或再分发授权；已核查的研究来源保留在对应模型和数据文档中。发布前必须完成[发布检查](docs/product/releases.md#发布前要求)。
