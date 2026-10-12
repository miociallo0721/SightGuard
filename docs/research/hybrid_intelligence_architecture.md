# Hybrid Intelligence 架构与边界

本次先通过两种真实模型的 P0 推理，再实施 P1–P4；完整验证边界见验证记录。
实际提醒控制器始终为现有 Rule Engine，AI 仅作影子研究输出，不影响 AlertEvent 或空间音频。
AGENTS.md、现有风险评分/事件状态机及冻结 Phase 3 产物不修改。

## 已核实的模型来源

- Structured 官方项目：[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，
  发布者 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)。
  类型化 choice/score/noul 判断接口，不能按生成聊天 JSON 的模型调用。
- Vision：[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision)，
  代码 [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)。
  201M 版本基于裁剪的 SmolVLM，支持原图与可选文本；它是独立研究分支。
  代码 Apache-2.0，权重 CC-BY-NC-SA-4.0，未来发布需审查非商业和相同方式共享条件。

两套上游发行包均使用 Python 包名 laya；不能在同一环境内依次安装覆盖。
本机主 PyTorch 为 CPU 版，YOLO 保持原环境。辅助模型准备在 models/hybrid/ 下的独立
虚拟环境，权重/下载缓存/上游安装源码均不提交。隔离进程接口用于防止依赖冲突和本机
驱动异常传播到快速规则链路。固定代码提交、模型 revision、精度和实际设备记入运行产物。

## 实施约束

- 快速链路不等待语义推理；有界最新状态槽、限频、超时和模式代际隔离。
- 输入仅使用原始图像、检测数据和时序特征，排除最终规则等级与实际提醒。
- 输出保留原始类型化结果；风险候选是研究标签，原始得分不是碰撞概率。
- 对齐 session/frame/timestamp/event，失效/过期/事件消失的结果不能参与实时融合。
- 规则与 AI 标签作并列比较，融合只使用可解释研究基线，不混合不同含义的原始得分。
- 加载/卸载与切模式不重新启动摄像头、YOLO 或事件引擎。只允许一个辅助模型占用GPU。
- 研究数据写入新的 outputs/hybrid/session-*，与冻结实验目录隔离。
- 本次不训练、不量化 Vision、不做 Android；后续 ONNX/LiteRT/NCNN 兼容性需单独实验。


## P0 版本与精度核查

Structured 模型固定 revision `7b928d828b7b0e022f929d9bd2e44165aa270148`；
官方代码 v0.4.2 固定提交 `68804629e8ccd9d616d48a40e87de9fabbeae069`。
该模型仓库没有 .onnx 成品，INT8 需要运行官方导出器的本地实验；
不能把它标成已下载的官方量化权重。上游动态 MatMul QInt8 的支持目标是 CPU ORT，
CUDA Execution Provider 兼容性与量化后的判断一致性必须另行实测。
Vision 固定 revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`，
权重文件为 F32 safetensors；FP16 是本次加载精度选项，不是发布者的 INT8/INT4 成品。

## 实际接口与运行方式

`DecisionProvider` 定义 load/infer/unload；RuleDecisionProvider 仅投影已计算的 StepResult，
不建立第二个风险引擎。ProcessDecisionProvider 使用 JSONL stdin/stdout，stderr单独保存。
一个语义线程拥有模型生命周期；选择新模式只递增 generation、取消旧任务并通知该线程。
旧进程退出后再加载新进程，不同时运行两套辅助模型。超时/取消会终止本任务拥有的进程；
错误只进入影子诊断，不反馈到摄像头或 AlertEvent。Rule模式完全不等待辅助加载。

Structured实际调用：`ONNXAgent(local_dir, onnx_path=local_file).system_one(state, questions)`；
官方动态MatMul QInt8在CPUExecutionProvider通过实测。ONNXAgent无公开线程参数，host释放
默认session再创建2线程session，避免同时常驻两份会话。FP32 eager备用配置调用
`laya.load(local_dir, device='cpu', compile=False, backend='eager')`。
Vision实际调用：`laya.load_vlm(local_dir, device='cpu', dtype='fp32')` 后
`agent.predict({'image': PIL.Image, 'note': context}, questions, n_permutations=1, batch_size=1)`。
Structured的criteria为标签字典，Vision为候选文本列表，解析分别读取真实choice。
配置中的FP16/CUDA路径保留但尚未实机验证；上游dtype选择器只处理bf16/fp32，host的
FP16分支显式同步模型与预处理dtype，仍需独立兼容性/输出实验。

辅助输入只选最高关注区域关联的观察事件及有限上下文目标，不用最高规则风险选目标。
检测、位置、观察持续时间、运动可靠性/变化率和感知限制保留；语义副本浮点四舍五入到
4位小数以限制token开销，完整精度保留在规则日志。每个输出关联同一来源frame/event；
比较中的Rule等级来自该输入帧，不拿当前帧等级与旧模型结果混用。
默认1Hz、8秒推理超时、90秒加载超时、2秒有效期；CPU超时/过期是实际限制，不能放宽
有效期后宣称低延迟。candidate_score保留上游选项置信/比较得分，不跨模型做加权概率。
conservative_max只取合法候选标签的较高等级，consensus只接受同等级，两者仅为研究基线。

网页默认7860，复用原产品FastAPI控制器，cookie所有权约束不变；原Gradio研究台7861
仍使用既有功能。POST `/api/live/intelligence` 只接受mode/auxiliary/action三个枚举字段，
不接受浏览器任意Python程序或模型路径。选择模式先卸载，明确点击加载后才启动模型。

## 可复现环境与权重

主环境继续使用uv.lock，不安装两个冲突的上游laya包。辅助包冻结列表在
`configs/hybrid-structured-requirements.txt`、`configs/hybrid-vision-requirements.txt`。
运行 `scripts/prepare_hybrid_environment.ps1 -Model laya` 或 `-Model vision` 可明确安装运行时，
该脚本不会下载权重。Vision固定代码commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`；
CPU环境通过.pth只读复用项目.venv的CPU torch，不是自包含环境；复制项目后需重跑脚本
更新.pth。Linux/macOS需手动建立等效环境并修改hybrid.yaml解释器路径。

权重准备属于用户主动操作，固定模型来源：
Structured使用huggingface_hub.snapshot_download(repo_id='convaiinnovations/laya',
revision='7b928d828b7b0e022f929d9bd2e44165aa270148', local_dir=本地目录)。
Vision使用相同方法指定thaitea/laya-vision及上文revision；它不是普通SmolVLM替代权重。
网络Xet下载曾停滞，本机固定revision的HTTPS Range分段下载后验证完整大小和SHA。
Structured官方导出器地址固定为
`https://raw.githubusercontent.com/NandhaKishorM/laya/68804629e8ccd9d616d48a40e87de9fabbeae069/scripts/export_onnx.py`，
调用 `--model 本地目录 --output 本地目录/laya.onnx --quantize`；产物laya.int8.onnx。
需要额外磁盘空间容纳FP32 external data与量化临时文件，均放models/hybrid/tmp。

本机host要求本地来源manifest与配置model_id/revision/权重hash一致：Structured的
`model_manifest.json`含model_id/huggingface_revision/source_checkpoint_sha256，ONNX时增加
onnx_export.sha256；Vision的`sightguard_model_manifest.json`含model_id/revision/
model_safetensors_sha256。P0脚本另行校验并输出版本/原始结果；不会自动下载或创建虚假manifest。
本机实际SHA见验证记录，重导出得到不同哈希时应更新实际manifest，不照抄本机哈希。

日志包括metadata/inputs/rule_decisions/laya_decisions/vision_decisions/comparisons/performance，
metadata保存YOLO及辅助权重hash、配置、时间来源、运行环境、代码commit、许可证。
规则日志是语义采样源帧；图像仅在retain_images=true时保留原始PNG，默认只存像素hash。
读取DecisionResult.from_json可重放诊断结果；未保存图像时不能重新运行Vision输入。

## 后续移动端障碍

JSON契约不依赖UI/PyTorch，可作为Kotlin移植起点。Structured已有CPU ONNX实测，但其
tokenizer、动态shape、量化MatMul与运行时选项需要在Android ORT单独验证。
Vision仍是PyTorch的裁剪VLM/视觉编码与typed decision head，尚无本项目验证过的
ONNX/LiteRT/NCNN转换产物；视觉patch预处理、attention mask、readout与决策头都需要
分别导出/校验，不能宣称已支持NCNN/LiteRT或可直接移植。还需处理非商业权重许可证，
以及Windows隔离子进程/Python包生命周期在原生应用中的替代实现。
