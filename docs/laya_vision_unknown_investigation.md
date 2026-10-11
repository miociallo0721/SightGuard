# Laya Vision UNKNOWN 调查

调查日期：2026-10-11。固定模型 `thaitea/laya-vision` revision
`f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`，代码 revision
`9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`。不修改生产问题、风险算法、融合条件或提醒行为。

## UNKNOWN 的确定来源

上游 `laya/vlm.py::VLMAgent.predict` 是类型化分类接口，接收 `image`/可选文本和
`questions`；choice 的 criteria 文本列表转成候选字典，输出候选概率及 argmax 对应的文本。
它没有为当前 choice 任务自动追加 UNKNOWN，也没有由某个置信阈值自动转换成 UNKNOWN。

SightGuard 的 `hybrid/schema.py::CRITERIA` 显式包含
“Insufficient or unclear evidence to select a concern category.”。
此前固定输入的原始输出选择了该文本，概率约 0.4512，confidence 约 0.0807；
项目解析器将原文映射为 UNKNOWN，候选风险标签设为 None。
因此它是**模型选中了项目提供的弃权选项**，不是解析失败、超时或 GPU 回退导致的兜底结果。
重复同一输入 30 次不是 30 个独立场景。

上游的 `action.act_probability` 是独立 action head 输出，不能据此把 UNKNOWN 改为风险标签。
confidence 为候选分布统计，不能解释为碰撞概率。上游 API 和教程见固定 checkout 的
`site-docs/tutorials/first-prediction.md`、`laya/vlm.py`；[官方代码仓库](https://github.com/r33drichards/laya-vision)。

## 固定小测试集与任务

脚本 `scripts/investigate_vision_unknown.py` 不下载模型或图像，沿用实际预处理器。
保存 RGB PNG、像素 SHA、上下文、问题、原始 logits/概率、标准化结果、耗时和参数信息：

- A：单个红色球的自行绘制示意图。
- B：前方中央明显棕色箱子的自行绘制示意图。
- C：无前景物体的开阔地面示意图。
- D：同一箱子被大面积遮挡并模糊后的信息不足示意图。
- 上游随固定代码附带的 DJ 设备教程照片（记录来源/哈希，不另行下载）。
- 原 Benchmark 合成场景与原完整语义上下文。

前四项是确定性诊断插图，不能当成自然图像验证集。照片仅为本地教程诊断使用，未复制到 Git。

每个场景保留原风险 choice；另外测试前景颜色、物体存在性、中央下方占用描述。
教程照片还按上游示例测试 category choice、record noul 和 condition score。
新增问题只判断可见属性/占用，不把“存在物体”直接解释为碰撞危险。
没有删除 UNKNOWN、改变温度、降低阈值或强选风险标签。

## 实际发现

GPU 混合精度在六个输入的原风险任务上均选择 UNKNOWN，原 Benchmark 30 次亦如此。
但基础视觉任务能够产生合法的具体选择：红球颜色为 red、箱子颜色为 brown、箱子存在性为
“A foreground object is visible”，教程设备类别为 electronics。

同时存在明显错误与不一致：红球场景的存在性回答为无前景物体；空地与遮挡场景的中央占用
任务回答为有物体。红球与教程照片的中央占用任务选择了无法判断。这些“无法判断”文本是
各任务自有的弃权描述，不能仅因不等于风险 UNKNOWN 原文就算成有信息的候选。

六张图中有五张运行四个 choice 任务，教程额外一个 category，原场景仅原 risk：
共 22 个 choice 答案，其中 6 个为原风险 UNKNOWN；其余 16 个是其他任务合法选择，
其中 2 个也是显式无法判断。因此具体非弃权 choice 为 14 个，**有效非 UNKNOWN 风险候选为 0**。
另有教程的 noul 和 score 各 1 项，它们不是离散风险 choice。

这说明真实模型/API 能完成具体视觉分类，并非所有任务都弃权；“general concern” 风险选项
缺少可观察的明确标准，且输入只是插图/静态图，不提供可靠碰撞证据。
问题语义适配和输入领域是需要继续调查的因素，不能从本小样本断言唯一因果。
没有发现图像丢失或 UNKNOWN 标签映射错误；正常编码不等于预处理最优或模型能力已验证。

## 设备与输出验证

GPU 运行时 hook 记录视觉塔输入/输出为 cuda:0 FP16，scorer 输入/输出为 cuda:0 FP32；
decision model 也记录关联输入/输出。Profiler 单次原场景前向记录 1225 个 CUDA 设备事件，
原 trace 保存于 `unknown-cuda-2026-10-11/cuda_trace.json`。
这与参数位置共同证明实际 GPU 前向，不把 CPU 预处理或最终 `.cpu()` 输出当作模型回退。

原始结果分别保存于 `outputs/hybrid/gpu-benchmark/unknown-cuda-2026-10-11/`、
`unknown-cpu-2026-10-11/`，不覆盖历史记录。CPU/GPU 同输入比较与性能更新见
[GPU 可行性研究记录](hybrid_gpu_feasibility.md)。

CPU FP32 同样六个风险任务全部 UNKNOWN；CPU/GPU 的 22 个 choice 选择全部一致，
所有输出数值有限，没有 NaN/Inf。各任务原始概率的最大绝对差为 0.0032（已经过上游
四位小数舍入），没有发现问题被截断的字段。对照已断言图像像素、候选问题及上下文完全一致，
写入 `2026-10-11-environment/comparison.json`。这只验证小样本数值一致性，不证明所有输入稳定。

## 复现与研究结论

```powershell
./models/hybrid/env-vision-gpu/Scripts/python.exe scripts/investigate_vision_unknown.py --device cuda --precision fp16 --output outputs/hybrid/gpu-benchmark/unknown-cuda-new
./models/hybrid/env-vision-gpu/Scripts/python.exe scripts/investigate_vision_unknown.py --device cpu --precision fp32 --output outputs/hybrid/gpu-benchmark/unknown-cpu-new
```

脚本内部将缓存/TEMP 路径设到 D 盘并启用离线模式，新目录必须未存在。CPU FP32 对照使用
同一个 CUDA 环境中的 CPU 运算以统一 Torch/依赖版本；原 CPU 环境另由回归验证保留。

目前有证据支持将 Vision 继续用于具体视觉属性与任务语义的旁路研究；没有证据支持让其
接管 SightGuard 风险提醒。下一步需清晰的可观察任务定义、自然场景独立标注，以及任务
预测与事件风险的独立验证。推理成功、输出合法、没有弃权、符合任务预期和风险数据集正确
必须分别衡量。
