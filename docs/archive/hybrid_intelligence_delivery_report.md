> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# SightGuard Hybrid Intelligence 阶段交付报告

报告日期：2026年10月10日。项目目录：`D:\program\SightGuard`。

本阶段建立了可运行的多模型影子研究原型：两种辅助模型均已完成真实本地CPU推理，并接入
现有实时摄像头与产品网页。系统能够保存规则、辅助模型及实验性融合的研究结果。
**真实提醒与空间音频始终由既有Rule Engine控制。**

这是一项工程与模型可行性验证，尚不是经过独立数据集验证的混合风险算法。
尤其是CPU上的Laya Vision实时延迟超过默认结果有效期，当前主要支持旁路诊断与研究记录。

## 1. 开发范围与完成情况

本次沿用YOLO26n、BoT-SORT、Phase 2风险评分、事件状态机、AlertEvent及Spatial音频实现，
未改动这些算法的提醒条件，未修改AGENTS.md，未覆盖Phase 3冻结实验产物。

- P0：核查模型来源、许可证、真实API与权重格式；先完成真实推理，再启动后续集成。
- P1：实现DecisionProvider、结构化输入、真实输出解析、加载/推理/卸载生命周期。
- P2：实现实时影子通道、容量一最新输入槽、限频、超时、取消与旧模式结果隔离。
- P3：实现同源帧/事件比较与两种可配置标签融合基线，拒绝无效、过期及不可比较的候选。
- P4：复用产品网页，增加四种研究模式、模型管理、真实状态、风险对照及性能统计。
- P5：完成自动化回归、真实模型集成、Xiaomi摄像头验证、网页交互验证、构建和本地提交。

四种模式为Rule、Laya Structured、Laya Vision与Hybrid。Hybrid支持Rule+Laya或Rule+Vision，
只加载选定的一种辅助模型。模式切换不重新初始化摄像头、YOLO或现有风险事件。
切换后需要明确点击加载，不会打开网页就自动下载或启动模型。

## 2. 实际模型与运行环境

本机为Windows、Python 3.12、RTX 2050 Laptop 4GB。主环境PyTorch为2.14.1+cpu，
CUDA不可用。辅助模型使用隔离环境/子进程，避免两套同名laya包互相覆盖。

### Laya Structured

来源：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)，代码为
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) v0.4.2，许可证Apache-2.0。
模型revision固定为`7b928d828b7b0e022f929d9bd2e44165aa270148`。

FP32 eager真实推理已通过。默认接入版本为本地运行官方导出器生成的动态MatMul
per-tensor QInt8 ONNX，使用ONNX Runtime CPUExecutionProvider；该ONNX不是官方仓库中的
预制量化成品。产物598,403,188字节，加载与推理均通过真实验证。

实际调用为`ONNXAgent(...).system_one(state, questions)`，输出choice及原始候选得分，
不按聊天模型调用，也不将得分解释为碰撞概率。

### Laya Vision

来源：[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision)，代码为
[r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)，固定代码提交
`9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`。
模型revision为`f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`。

实际模型201,161,347参数，完整FP32 safetensors为804,696,880字节；没有用普通SmolVLM
替代权重。实际CPU FP32加载与`agent.predict(image/context, questions)`均通过。
代码Apache-2.0，权重CC-BY-NC-SA-4.0；未来商业化或应用分发需要审查使用许可。

**GPU FP16、Vision INT8/INT4及显存性能均未验证。**显存统计保留null，不把CPU运行写成
“GPU占用0MB”。Vision环境通过.pth只读复用主环境CPU torch，迁移机器需重新配置该路径。

## 3. 双速率与数据边界

快速链路继续执行摄像头→YOLO/跟踪→规则风险→真实提醒/空间音频，不等待辅助模型。
语义线程在旁路按配置采样；输入积压时覆盖尚未处理的旧状态。

辅助模型仅接收检测、位置、观察时间、运动变化/可靠性及感知限制。Vision使用对应的
原始摄像头图像。输入排除最终规则风险标签，避免将规则答案交给模型再进行“独立对照”。
语义副本浮点文本保留4位小数，原规则计算与采样日志仍保留完整精度。

结果包含版本、session/frame/timestamp/event/model、原始输出、候选标签/得分、耗时、
有效性与错误。模式切换通过generation和取消事件隔离旧任务；超时终止所属辅助子进程。
目标消失、事件失效或源帧过期时，候选不能参与当前融合。

默认1Hz采样、8秒推理超时、90秒加载超时、2秒源帧有效期。
融合仅使用合法标签：conservative_max取较高研究等级，consensus接受一致等级。
没有混合不同模型的原始得分，也没有声称某组融合权重最优。

## 4. 实际性能与解释范围

短合成输入P0实测：Laya CPU INT8三次推理约184.3/180.8/181.8ms；
Vision CPU FP32最后一次合成图像推理约852.6ms。输入不同，这些数字不是统一Benchmark，
更不能替代完整实时链路的延迟。

最终网页会话`session-a49f2df95ed749a880290e6afbef57b1`保存的实际统计如下：

- Laya：90次调用，P50约1547ms、P95约1610ms，加载约14.45秒。
- Vision：15次调用，P50约3829ms、P95约3937ms，加载约12.95秒。
- 两种辅助合计105次调用，0次超时；过期计数105，包含完成后随后失效的结果。
- 规则风险更新FPS滚动均值：辅助启用前约28.12，启用后约23.34。

前90条Laya记录中89条在完成时有效，但随后仍可能因源帧年龄超过2秒而从当前比较中隐藏。
因此过期计数不能解释为105次模型推理失败，也不能把“完成时有效”当成持续有效。
Vision的实时CPU调用约3.8秒，超过默认有效期，通常不能产生有效的当前融合候选。
网页明确展示过期/UNKNOWN，不继续呈现旧判断为当前风险。

上述来自室内短会话及滚动统计，未进行独立控制变量实验、长期稳定性验证、道路试验或
使用者测试。没有独立ground truth，因此不报告准确率、召回率、误报率或安全性结论。

## 5. 验证结果

实际执行并通过：

- `uv run ruff check .`及`uv run ruff format --check .`。
- 默认pytest：306通过，4项需要显式本地权重的集成测试未启用。
- 显式启用本地YOLO和辅助模型后，最终全套pytest：**310通过，40.49秒**。
- 独立真实集成测试：2项YOLO与2项辅助子进程测试，合计4项通过。
- `uv run python scripts/verify_hybrid_live.py --camera 1 --seconds-per-mode 20`：
  Xiaomi640×480MJPG实际处理，Laya/Vision/Hybrid切换后返回Rule，同一规则引擎保持连续。
- 产品网页真实操作：摄像头启动、选择模式、主动加载、Loading/Running展示、卸载和停止。
- `uv build`成功；归档未包含模型、缓存或运行输出，wheel包含model_host。
- PowerShell运行时准备脚本语法检查通过；未宣称已在另一台干净机器完整复现安装。

Mock测试覆盖异常、超时、卸载、旧代隔离、事件失效、会话隔离、输入/输出契约及提醒
不受辅助模型修改等路径。Mock的OOM测试不等于已执行真实GPU显存不足试验。

## 6. 交付文件与复现入口

核心文件位于`src/sightguard/hybrid/`，包含配置、输入、数据契约、Provider、子进程host与
异步Manager；`live/runtime.py`提供旁路接入；产品网页及服务层增加Risk Intelligence。

主要配套文件：

- `configs/hybrid.yaml`及两套辅助环境的固定依赖列表。
- `scripts/prepare_hybrid_environment.ps1`、两种模型P0验证脚本、实时验证脚本。
- `tests/test_hybrid.py`、`tests/test_hybrid_integration.py`及既有UI回归测试。
- [架构文档](../research/hybrid_intelligence_architecture.md)、[验证记录](hybrid_intelligence_validation.md)、README。

本地原始记录在`outputs/hybrid/p0/`与新的`outputs/hybrid/session-*`，均被Git忽略。
包括来源/配置/权重哈希、原始模型结果、规则采样帧、同源比较、错误及性能数据。
默认不保留原图；仅有像素哈希时不能重新执行Vision图像推理，需精确复现时明确开启
`retain_images`并管理图像数据。模型之间的一致率不得当作识别准确率。

启动命令：

```powershell
uv run sightguard ui --port 7860
```

在实时体验中选择Xiaomi摄像头，启动后选择研究模式并点击加载。原研究台仍为
`uv run sightguard ui --research --port 7861`；音频实验页使用7862。
验证结束后设备已释放，服务待机。报告编写时三个端口均返回HTTP200，摄像头未运行。

## 7. Git与磁盘管理

开发分支：`codex/hybrid-intelligence`。代码提交：`ccc0cbc`，
`feat: add hybrid intelligence shadow research`。共29个文件变更，未推送远端，
未创建正式发布标签；包版本仍为0.4.0。本报告为该提交之后新增的文档。

收到用户磁盘约束后，本阶段临时文件、下载缓存、模型和输出均改用D盘。
此前一次ORT量化在C盘留下约1.69GB临时文件，目录为
`C:\Users\55043\AppData\Local\Temp\ort.quant.6o97348o`。
用户要求尝试清理后，重新检查确认只有3个本次量化文件，但删除再次被自动审批策略拒绝；
代理未执行删除，也未绕过限制。报告编写时复查，该目录已不存在；未核实由何种操作清理，
也未测量实际释放空间。共享uv缓存未由代理清理。

## 8. 未完成事项与后续重点

首先应验证RTX2050上的辅助模型CUDA/FP16兼容性、真实延迟和显存，解决Vision候选普遍
过期的问题；不能通过延长有效期来宣称低延迟。然后建立独立标注数据集，评估候选标签、
校准与融合策略，量化误报/漏报及提醒效果，之后才考虑扩大实际提醒控制权限。

Android仍需验证Structured的tokenizer、动态shape与量化算子；Vision尚无本项目验证过的
ONNX/LiteRT/NCNN转换产物。视觉预处理、attention/readout与决策头需要分别校验，
并替代Windows/Python的进程管理及依赖隔离方式。本次未开发Android、导航或模型训练。

现阶段可交付的是**真实模型可运行、实时规则链路保持独立、网页可观察并记录模型差异的
研究工程基础**。它补充盲杖和导盲犬，不提供可靠碰撞预测或安全保证。
