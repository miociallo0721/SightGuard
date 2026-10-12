> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Hybrid Intelligence 验证记录

## 初始环境与已有功能

2026-10-10，在 `codex/hybrid-intelligence` 分支开展 P0。
基线提交为 `76248f3`，现有规则/音频功能沿用；主环境 Python 3.12、PyTorch 2.14.1+cpu。
`torch.cuda.is_available()` 为 false，而 nvidia-smi 能识别 RTX 2050、4096 MiB、驱动 616.56。
GPU存在不等于当前PyTorch环境可执行CUDA；辅助环境单独准备，主环境不替换。

实机规则基线命令：`uv run sightguard live --camera 1 --device cpu --duration 8`。
读取 640×480 MJPG，末次日志 processed=82、objects=2、窗口 risk_fps=26.43、processing_ms=32.0。
初始化期间出现感知过期提醒，之后产生首次风险事件提醒，退出资源正常释放。
只是一段室内短运行，不作为准确率、安全保证或全面性能Benchmark。
本地原始日志与环境信息位于 `outputs/hybrid/p0/`（Git忽略）。

## 模型P0与集成门槛

用户已明确允许下载两种固定版本权重用于真实推理测试。
官方Laya与Vision的API、许可证、权重格式已核查，见架构文档。
先完成Structured真实PyTorch推理，再开始P1–P4；之后INT8与Vision真实推理均成功。
未完成真实推理时不能把Mock输出、成功下载或成功import当作模型运行通过。
模型下载错误、加载失败、算子/驱动不兼容应保留原始原因。

## 实际模型与计时

Structured源权重842,609,210B，SHA256
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`；
官方本地eager加载后为FP32，合成样例load12,267ms、首次system_one644ms。
本地官方导出器生成MatMul动态per-tensor QInt8 ONNX598,403,188B，SHA256
`321bd438e66b3b413036d79fea2923143b7f7b599a00965c74d52482d83c6c5a`。
离线CPUExecutionProvider最终一次load7,711ms，三次合成样例推理184.258/180.841/181.846ms。
eager与INT8该样例都选none，但选项得分明显不同，不能据此声称量化精度验证通过。
上游出现choice11+校准温度回退提示；本题4选项依然不能称为已校准风险概率。
正式子进程适配器也真跑过同源本地INT8，另一短输入测得load10,195ms、调用146ms。

Vision为201,161,347参数完整决策模型，804,696,880B FP32 safetensors，SHA256
`31220803f38b44b8b1649b207c632612c9a35dee0da2ef1ea35d21e4b424be68`。
三个离线CPU合成图像验证中，最后一次load3,976ms、predict853ms，首轮predict990ms。
正式子进程适配器的黑色测试图load9,108ms、调用1,434ms，实际答案为UNKNOWN。
这些输入不同，不能混为统一Benchmark；均只验证能运行，并不验证风险分类准确率。
主环境与Vision隔离环境复用CPU torch，当前CUDA False；RTX2050存在但GPUFP16未验证。
尝试准备CUDA运行时的大型wheel后因下载成本停止，未声称GPU成功或显存配置最优。
实际CPU显存峰值为null，不填虚假的0MB GPU推理数据。

本机详细原始记录：`outputs/hybrid/p0/laya-structured-pytorch-smoke.json`、
`laya-structured-onnx-int8-final.json`、`vision_p0_20261010T141845Z.json`、
`host-laya.json`、`host-vision.json`。均在Git忽略目录；P0脚本可使用本地权重复测。

## 实时与网页

实际运行 `uv run python scripts/verify_hybrid_live.py --camera 1 --seconds-per-mode 20`，
Xiaomi640×480MJPG经真实YOLO、原Phase2、Laya/Vision/Hybrid旁路，随后返回Rule模式。
三个阶段的AssessmentEngine为同一实例，无摄像头/YOLO重建；JSON有实际原始模型回答。
末次风险FPS约25.24/23.40/23.25（各阶段rolling样本），整体0次辅助超时、5次过期计数。
该轮真实结构化输入较长，辅助调用约1.5–3.7秒，部分结果过期、不参与融合；
不是每次推理都有有效风险候选，也不是长期稳定性/道路试验。日志`live-smoke.json`。
随后语义输入精简冗余特征并将浮点文本保留4位小数，原规则记录与评分精度不变。
不能把P0短合成输入的~182ms直接标作完整实时推理延迟。

网页通过现有7860产品服务进行真实按钮验证：Xiaomi启动、选择模式、主动加载、
状态Loading→Running、CPU/INT8、实际延迟与过期原因、Rule/AI对比、卸载/停止。
网页始终显示Actual Alert Controller=Rule Engine。已有浏览器旧tab输入曾超时，
在同一浏览器新tab恢复；没有用浏览器注入伪造模型结果。
最终精简输入网页会话 `session-a49f2df95ed749a880290e6afbef57b1` 中，
前90条Laya结果有89条在完成时valid；随后仍会因2秒源帧有效期到期而隐藏，
“完成时有效”不等于“稍后显示时有效”。Laya网页P50约1.55秒，Vision约3.83秒，
后者超过默认有效期，因此本机CPU实时Vision通常不能提供有效融合候选。
同会话加载前风险FPS约28.6，辅助启用后约23；不是独立统计对照或普遍性能保证。
用户可从原始结果研究问题，但不能为了持续显示候选而把过期结果当作当前判断。
最终页面截图保存在D盘 `outputs/hybrid/p0/web-hybrid.png`、`web-vision.png`，
摄像头/辅助模型验证结束后释放，7860服务保留待机。

## 自动化与构建

本次 `uv run ruff check .`、`uv run ruff format --check .` 通过；
`uv run pytest -q` 默认306通过、4项显式真实集成未启用；
设置SIGHTGUARD_TEST_MODEL为本地yolo26n.pt和SIGHTGUARD_TEST_HYBRID=1后，
最终全套310通过（40.49秒），包含2项真实YOLO与2项真实辅助子进程测试。
单独真实集成命令 `uv run pytest tests/test_integration.py tests/test_hybrid_integration.py -q`
4通过。最终输入精简后的定向33项也通过。Mock的OOM/取消/隔离测试不等于实机CUDAOOM验证。
`uv build`成功，产物sightguard-0.4.0 sdist/wheel；检查归档仅包含models/outputs的gitkeep，
不包含模型、缓存或摄像头输出，wheel包含独立model_host。此研究增量不创建正式发布标签。

## 限制与磁盘路径

所有新增运行临时目录、缓存、权重与实验输出已显式指向D盘。用户提出此约束之前，
uv安装用过C盘共享cache；首次ORT量化在C盘临时目录遇到空间不足，后改D盘重跑成功。
本次可识别遗留 `C:\Users\55043\AppData\Local\Temp\ort.quant.6o97348o` 约1.69GB，
自动审批拒绝删除该目录，因此未删除，也未尝试绕过或清理共享缓存。
已有C盘Python解释器为先前环境，只读取；新脚本不会在C盘安装中间模型产物。

无独立ground truth，不提供召回/误报/准确率或安全结论；Vision许可证为非商业共享条件。
GPUFP16、量化全面一致性、真实道路与使用者验证、Android算子转换、长期运行、
GPU并行占用及安全评估仍需后续独立工作。本次保持真实提醒条件与原空间音频模块不变。
