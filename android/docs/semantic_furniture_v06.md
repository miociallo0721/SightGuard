# Android 0.6：家具语义参与环境感知

## 开发基线与验证约定

基线：`5a7b471fb48b8632bbc20a9506317e1f9fb42778`，分支 `codex/android-semantic-furniture-v06`。
基线 ARCore PR #9 的测距门禁、正式风险评分和冻结 PC benchmark 保持不变。
此轮为 **物体存在提醒**，不判断人体净空、桌底能否通行或物体距离。

在真实模型评测前固定如下小样本试运行门禁，不根据预测结果挑选图片或降低置信度：

- 数据：官方 ADEChallengeData2016 的 validation 分割标注；先扫描前 160 张标注，按标注覆盖率选取每个家具组前 4 张覆盖率至少 1.5% 的图片，以及前 8 张没有家具组超过该覆盖率的图片。
- 这些“无大面积家具”样本可能仍包含小家具；每类统计保留全部真实标注，不将它们虚称严格空场景。
- 新类别默认保持概览/研究可见。候选自动提醒组需要至少 4 张该组标注覆盖率达到 1.5% 的样本，至少一半产生连续语义事件，并且聚合像素 precision ≥ 0.60、IoU ≥ 0.30。
- 不允许在该组完全没有标注像素的场景产生该组自动事件；保留混淆与失败例。实际音频需要至少一次该类别真实模型事件对应的 onStart。
- 此门禁仅支持有限数据集上的功能试运行，不代表量产准确率、跨域泛化或道路安全验证；未满足的类别维持概览，不为了数量启用。
- 评测所有新类别时，instrumentation 可临时打开事件组以记录命中和误报；不会改写正式应用配置。
- 同一静态图片的重复推理仅验证连续确认代码和真实推理时效，不等同真实运动、CameraX 或接近验证。

## 架构

`EnvironmentTaxonomy` 统一 ADE20K 原始 ID、中文名称与事件组。桌子（15/33/64）、椅子（19/30/31/69/75/110）、柜子（10/35）、沙发（23）、床（7）、架子（24）、柜台/台面（45/70）、冰箱（50）保留原始类别来源。

`SemanticStructureMeasure` 复用连通区域、关注区有效面积、置信度及连续观测；不同家具区域保持独立身份。
`SemanticEnvironmentEngine` 继续负责确认、恢复、过期、去重和结束，隐藏研究图层不关闭语义提醒。

`EnvironmentSpeechDeduplicator` 仅接受真正开始播放且明确说出类别的 YOLO 提示，以类别族、短时窗口、会话和区域重叠抑制重复环境提示。通用“目标”音频不能冒充“桌子”已播报。语义存在提示不会阻断正式风险或风险升级。

空间网格增加家具语义支持率及原始类别；仅应用端启用，默认配置保持 PC 对照兼容。它表示图像表面占用，不证明米制净空。缺失、冲突和未知证据继续保留。

中文音频新增 8 组 × 3 方向，复用原有 TTS/本地音频、静音、抢占、过期取消和 onStart 回执。环境概览统一家具子类名称，避免与 YOLO 重复罗列。

## 可复现命令

```powershell
. ./android/tools/task_environment.ps1
python android/tools/fetch_furniture_validation.py
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest --console=plain
adb -s DEVICE install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s DEVICE install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
python android/tools/push_furniture_fixtures.py --serial DEVICE
adb -s DEVICE shell am instrument -w -r -e class org.sightguard.app.FurnitureDatasetInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

数据、模型和设备原始日志保留在本地忽略目录，提交聚合报告及文件哈希。最终结果见下文及 `evidence/semantic-furniture-v06/`。

## 扩展验证与保留失败

初始样本仅含 1 个按配额选中的沙发场景，没有达到面积要求的冰箱场景。为继续检查泛化与误报，另外固定扫描第 161–480 张 validation 标注，每组最多 8 张、16 张无大面积家具图片；与初始图片不重叠，不根据模型预测选图。保持原阈值，单独报告两批结果。

已发现并保留的工程失败：

- 首次真机模型测试触发映射门禁：官方映射 ID 7 的 rawLabel 为 `bed `（带尾随空格）。修正注册表后，原有模型数值对照及双线程 softmax 对照通过。
- 真机共享存储夹拒绝应用读取 ADB 写入的测试文件。改为标准 `run-as` 调试身份将公开样本放入本应用私有目录，不更改存储权限。
- 红米锁屏长批次测试被系统冻结，主线程和测试线程的等待状态为 `do_freezer_trap`。停止该批次，改用独立短批次；不关闭后台保护，不解锁。短批次不能当作真机连续运行验收。
- `debuggerd` 只允许 root，因此跳过该诊断途径；未执行 root 或系统修改。

## 同一家具族置信度对照

首批原始类别置信度评测暴露出椅子/柜子像素识别与事件确认之间的缺口。
加入可关闭的 `furniture_family_confidence` 对照：保留原始 argmax、原始类别置信度及 `.35` 模型输出门槛，对家具事件使用同一族真实 softmax 概率之和。例如椅子族为 chair/armchair/seat/bench/swivel chair/stool 概率之和。
这表示互斥原始类别的并集支持，不是把模型置信度乘系数；不会聚合 table 与 counter，也不会给墙面、台阶或其他类别提高置信度。事件面积、像素数、`.60` 置信度、3 次确认、500ms 时效门禁均不降低。

报告同时保留 `raw_class_confidence`、`confidence_basis`。通过真实模型 logits 数值对照验证概率和，保留原始规则结果；是否采用新规则依照独立后续样本的误报与命中结果确定。像素级分割统计仍使用原始 argmax 输出。

## 最终识别结果与启用范围

共 88 张官方 **validation** 图片：初始 28 张，后续不重叠的 60 张。它们来自与训练相同的 ADE20K 场景域，不是独立实地泛化验证，更没有使用训练集成功率代替验证。后续子集的选择在后续推理之前固定，但整体结果仍属于开发期小样本评估。

原 256 输入与家具族概率对照的合并结果，没有新家具组通过全部预设门槛。另行导出 **相同冻结权重的 384 输入**，保持预处理和原始输出类别，得到以下 Android AVD 真实模型结果。事件列为 GT 面积 ≥1.5% 的图片中产生有效连续事件的数量；每张图重复真实推理 4 次，不代表 4 个独立场景。

| 类别 | 有事件 / 阳性图 | 像素 precision | IoU | 结论 |
|---|---:|---:|---:|---|
| 桌子 | 8 / 27 | 0.529 | 0.321 | 概览/研究；精度和事件覆盖不足 |
| 椅子 | 11 / 27 | 0.738 | 0.484 | 概览/研究；事件覆盖不足 |
| 柜子 | 16 / 29 | 0.731 | 0.561 | 概览/研究；`ADE_val_00000164` 无对应 GT 却产生事件 |
| 沙发 | 2 / 10 | 0.576 | 0.322 | 概览/研究；精度和覆盖不足 |
| 床 | 15 / 21 | 0.826 | 0.765 | 384 模型下允许存在提醒；仍有 6 张漏报 |
| 架子 | 3 / 13 | 0.509 | 0.339 | 概览/研究；两张零 GT 误事件保留 |
| 柜台/台面 | 2 / 11 | 0.673 | 0.236 | 概览/研究；IoU 和覆盖不足 |
| 冰箱 | 6 / 8 | 0.805 | 0.575 | 384 模型下允许存在提醒；样本量尤其有限 |

像素统计采用归一化 128×128 像素中心采样，忽略标注 0；低置信度/缺失模型像素仍计入漏检。它不是官方全量 ADE20K mIoU 排名。初始、后续、合并及逐图统计全部保存，不能只引用床的成功例。

正式配置新增自动提示只有 **床、冰箱**；沿用墙面、楼梯、杆柱和栏杆/扶手。台阶继续未启用。8 类家具均可参与主动环境描述、研究证据和图像网格占用，但不宣称具体实例数量、人体净空或桌下可通行。

## 调度、模型契约及回退

- `SegFormerModels` 注册 256/384 静态形状和 SHA；`SpatialDetector` 校验实际资产、输入输出尺寸和 ADE20K 类别映射。
- `SemanticResolutionPolicy` 在连续两次源帧年龄 >400ms、明显主链负载或模型失败时，单会话退回 256。它是资源策略；实际消费门禁仍为 500ms，未延长。
- 同一辅助工作线程先关闭旧 Session，再建立新 Session，不并行持有两个 SegFormer；原有热状态、低内存和 YOLO 优先调度继续生效。
- 床/冰箱资格绑定 384 模型 SHA。切换模型后重新建立连续观测，保留已经播报的区域身份；256 输出不能借用 384 的验证资格。
- AVD release CameraX 实测前两次 384 源帧年龄 1717/691ms，**确实失败并触发 256 降级**。这不是 384 实时相机验收成功。之后基础 YOLO 与语义继续运行。
- 修复普通界面直接显示 `UNKNOWN_CHECKING` 等技术状态码的问题；显示中文受限原因，原始 ARCore 状态保留诊断。本轮没有改变 ARCore 几何、路径或语音门禁。

AVD 虚拟场景 UI 对照：[修复前](evidence/semantic-furniture-v06/ui-technical-status-before.png)、[中文状态修复后](evidence/semantic-furniture-v06/ui-chinese-status-after.png)。这些截图不来自用户私人摄像头。

## Android 与声音证据

红米 Turbo 4 Pro（25053RT47C，Android 16/API 36）通过 ADB 保持连接，未解锁或更改后台保护：

- 256 原始 softmax、家具族概率和 384 PyTorch 同张量对照共 3 项最终测试通过；384 最大绝对 logit 差约 `0.00012207`。
- 384 独立 Bitmap 推理记录 319/241/235ms，加载 132ms；采样 PSS 从 113947 到最高 457117KB，包含测试张量/输出缓冲，不能直接解释成纯模型额外内存。
- 正式配置实际床事件：第一帧 514ms 过期，后续 273/252/244ms 新鲜推理完成确认；中文音频出现 1 次真实 `onStart`。这是标注静态图片推理，**不是手机 CameraX 场景**。
- 39 条方向环境固定音频均记录实际 `onStart` 和 completed；本轮不声称有人逐条听辨。
- 较早长批次/连续批次记录到锁屏 HyperOS 的 `do_freezer_trap`。最终冰箱批次未完成，由主机 force-stop，日志因此显示 Process crashed；未取得该次冻结栈，不将其推定为模型崩溃。冰箱正式配置改由 AVD 验证，没有把未完成的真机批次计为成功。

AVD（API 36，2 vCPU、1536MB guest，16GB RAM 主机）完成：

- 28 张 raw256、28 张 family256、28 张 family384、60 张 family256、60 张 family384 的真实 ONNX/标注/事件评测，每次 4 次推理。统计没有删除失败图片。
- 正式配置冰箱实际 `onStart`；同一正式配置改用 256 的真实床图测试无床/冰箱事件，确认降级资格门禁。
- 384 真实墙面 Mask → 连续事件 → 中文 `onStart`/completed；持续墙面不重复、无墙面结束。使用公开静态照片，未冒充动态相机验证。
- YOLO/文件解码/跳转共 5 项回归；墙面与主界面/三轮 CameraX 启停共 4 项；TalkBack 实际服务、大字体 200%、横屏、窄屏、48dp 操作区和音频共存测试通过。自动状态更新未抢占测试中的可访问焦点。
- 三轮相机会话运行/释放 PSS：347362→191435、445275→191715、431335→191580KB；没有从这三轮短测推断长期无泄漏。
- release 虚拟相机日志中，首帧之后 YOLO 报告 FPS 中位数约 15.0、P95 18.5；降级后语义推理中位数 156ms、P95 279ms，源帧年龄中位数 220ms、P95 337ms，仍出现被丢弃的过期值。采样进程 PSS 最高约 397MiB。**不是红米 FPS，也不是同条件基线性能对照**。

工程检查：159 项 JVM 测试、5 项 Python 工具测试通过；相关 Ruff 检查通过；debug/test/release 构建成功。具体 APK、提交与安装信息由同目录交付记录保存。

## 模型、数据与可复现资料

- 原权重 SHA：`8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`。
- 256 ONNX：`85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f`。
- 384 ONNX：`49166600bda475ede43b413e8b145b735fe4532361fa663646e237f19e7dad69`，15,150,109 字节。PC 同输入 PyTorch/ONNX argmax 一致率 100%，最大绝对误差约 0.00010681。
- [OpenMMLab 原权重](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth)；[ADEChallenge 官方压缩包](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)。下载器仅选择标注定义的子集，检查 Range、ZIP CRC 和逐文件 SHA。
- [CSAIL benchmark 仓库 BSD-3-Clause](https://github.com/CSAILVision/sceneparsing/blob/master/LICENSE) 不应解释为所有来源照片的无限制再分发授权。本仓库提交哈希、标注统计和衍生评测，不提交照片/标注/权重。
- [NVIDIA SegFormer 原始许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) 含非商业研究/评估限制；MMSegmentation 框架 Apache-2.0 不单独证明权重及上游全部可商用。APK 附带两份许可与来源说明，当前交付仍为研究评估版本。

384 导出（单 CPU 线程，与当前 exporter 同一依赖环境）：

```powershell
python android/tools/research_segformer_resolution.py --checkpoint models/path_relevance/segformer_mit_b0_ade20k_160k.pth --image android/validation/validation.jpg --size 384 --output android/deliveries/semantic-furniture-v06/resolution384
Copy-Item android/deliveries/semantic-furniture-v06/resolution384/segformer_384.onnx android/app/src/main/assets/segformer_mit_b0_ade20k_384.onnx
python android/tools/fetch_furniture_validation.py --start-index 160 --scan-count 320 --per-group 8 --negative-count 16 --output datasets/imported/ade20k-furniture-v06-holdout
python android/tools/push_furniture_fixtures.py --serial DEVICE --fixtures datasets/imported/ade20k-furniture-v06-holdout
adb -s DEVICE shell am instrument -w -r -e family_confidence true -e semantic_size 384 -e class org.sightguard.app.FurnitureDatasetInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
# exec-out 以二进制/UTF-8 安全方式保存 files/furniture-dataset-actual.json，再执行：
python android/tools/summarize_furniture_validation.py report.json summary.json
```

数据集 instrumentation 默认为全类别研究评测；加入 `-e production_config true` 才测试正式启用配置。`case_start`/`case_count` 支持可追溯短批次。模型和样本必须先准备，默认 JVM/Python 单元测试不需要网络或 GPU。

## 仍需改进

1. 椅子、桌子等连续事件漏报和柜子/架子误报仍未解决。优先扩大独立真实图片与视频负例，研究局部高分辨率、实例分离和遮挡，不应统一降低阈值。
2. 后续已完成红米实际 ARCore + YOLO 并发观察：384 仅 6 次观测源帧年龄中位数 664ms，触发 3 次会话降级；256 的 502 次观测年龄中位数 395ms。实际处理 FPS 中位数 15.33。384 的静态图片成功不能视为实时家具验收通过，降级后床/冰箱自动提示关闭。参见 phone-live-summary.json。320 已有 PC/短程 Android 对照，但完整事件与实时验证尚未完成，未设为生产配置。
3. 384 在 AVD 相机负载下会降级。需要真机受控性能对照后再优化预处理缓冲或计算后端；不以陈旧帧延寿提高表面成功率。
4. 本轮家具进入概览、环境状态机和研究网格，但未重写 ARCore 局部候选的几何占用规则。图像家具表面不等于人体碰撞体积。
5. 原 ARCore 可用性首次异步检查可能暂时回退基础相机，后续重试策略仍需独立验收；本轮仅修复其中文状态呈现。
6. 商用发布前仍需完成模型/数据权利审查、无障碍用户实测及安全场景验证；没有因为 README 更清楚就将研究原型标记为成品。

## 后续检查点补充

本检查点保留 320 输入研究接口，但默认仍为既有 384/256 策略，320 不视为通过完整验收。手机截屏人工检查发现桌面显示器与笔记本混淆及桌子漏框；私人截图和原始设备日志仅在本地，不提交。AVD 尝试直接安装旧 versionCode=6 APK 到 versionCode=7 时返回 INSTALL_FAILED_VERSION_DOWNGRADE，即便使用 -d 也未通过。旧 APK 和源提交可回退构建，但不宣称旧包能够直接保留数据降级；未卸载应用或清除数据。
