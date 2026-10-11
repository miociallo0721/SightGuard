# Android v0.3 空间感知研究原型

本版在同一 APK 中保留 YOLO 主提醒链，并加入四个独立空间模块：Depth 相对深度、SegFormer 环境语义、Spatial Grid 影子融合、IMU 运动质量。三方向音频是既有提醒的输出增强。系统补充盲杖和导盲犬，不提供完整导航、主动绕障、可靠碰撞预测或安全保证。

## 四模块与模型契约

- **Depth**：`DepthDetector` 使用现有 PC 的 Depth Anything V2 Small HF checkpoint，revision `5426e4f0f36572d16453bbda7a8389317b1bef99`，checkpoint SHA256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`，Apache-2.0。导出 ONNX SHA256 `01013aabdd38909eff174b88c177ec583967b37d14041bab8f91d405c3a1760e`，opset 17；输入 `pixel_values` float32 NCHW `[1,3,280,280]`，输出 `predicted_depth` float32 NHW `[1,280,280]`，保留原生值。
- **环境语义**：`SpatialDetector.analyzeAll` 运行真实 SegFormer MiT-B0 ADE20K。ONNX SHA256 `85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f`；输入 `pixel_values` float32 `[1,3,256,256]`，输出 `class_logits` float32 `[1,150,64,64]`。全 150 个真实类别通道参与 argmax/softmax；显示及概览聚合是其子集，不是新增模型类别。
- **Spatial Grid**：`SpatialGridFusion` 在同一源帧的 15×15 图像网格中融合语义覆盖、YOLO bbox 重叠和相对深度统计。结果有 `GROUND/OBJECT/WALL/VEGETATION/CONFLICT/UNKNOWN` 及质量/失败原因、模型来源 SHA。缺失或帧键不匹配的来源保留 UNKNOWN/PARTIAL。`productionDecisionChanged=false`，不接入生产风险评分或语音状态机。
- **IMU**：`ImuMonitor` 注册设备实际存在的加速度计、陀螺仪、重力与 rotation vector，保留有界历史；运动质量由陀螺仪和姿态变化规则生成。没有加速度积分、位置估计或导航姿态。

YOLO26n ONNX SHA256 `e093d7e1805d63cc1b671c333ce0b48b5cd33f70f64ea16c9b5bd713e33242cf`；`images` float32 `[1,3,320,320]` 输入，`output0` `[1,300,6]` 输出。三个 ONNX 均作为同一 build 的 assets 打包。模型来源、参数及实际验证分别见 `main/assets/model_manifest.json`、`depth_manifest.json` 与 Python 导出报告；模型未准备完整时，编译成功不代表三模型可运行。

ADE20K 映射位于 `main/assets/ade20k_labels.json`：MMSegmentation v0.17.0 `ade.py` 的真实 150 标签，零基通道 0..149（标注背景通过 `reduce_zero_label=True` 去除）。源文件 SHA256 `5c2a2e53efa3591e4c3584ace7b23fef5c5c73213a02189ecd7a6188dfbaa20b`；加载时检查 150 项及顺序。树木的“上部/下部”仅按图像纵坐标划分；类别聚合外框可以包含不连通区域，不等同实例掩码。地面、门、楼梯、水面等均是视觉语义，不等同可通行性或风险结论。

## 几何、时间与未知状态

Depth/SegFormer 均先保持宽高比缩放，再居中 letterbox；RGB `/255` 后用 mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` 归一化，padding 的归一化值为零。结果通过对应 pad/resize 变换回源帧坐标。Python 参考采用 OpenCV `INTER_AREA`，Android bitmap 采用 bilinear；同张量测试验证的是序列化 float32 输入，不声称两套图像预处理逐像素相同。PC 原 DPT processor 保持宽高比并允许可变尺寸，也不同于 Android 固定方形 letterbox。YOLO 保持既有 320 输入和 RGB114 padding 契约。

Depth 数值是逐帧未校准相对逆深度，大值相对近、小值相对远；[官方论文 §5.2](https://arxiv.org/html/2406.09414v2#S5.SS2)说明模型预测 affine-invariant inverse depth。bbox 内缩采样是区域代理，非实例分割；分位关系不能转换成米或跨帧固定阈值。非有限值、低有效比例或缺少变化时返回不确定信息。

所有辅助结果携带 generation、epoch、frameIndex、源时间、尺寸、capture timestamp/clock、旋转等帧键，网格只接受同一键。UI 显示真实源快照，不把旧深度直接叠到新 YOLO 帧；停止、跳转、切换及过期结果被隔离。辅助快照 TTL 为 3 秒，文件暂停时允许精确当前帧保留。

IMU 对齐要求 Camera timestamp clock 为 `REALTIME`，有效 capture timestamp 与传感器时间窗口可比较；未证明时返回 UNKNOWN。传感器缺失、无效或过期为 LIMITED，快速转动为 RAPID/DEGRADED。当前设备运动诊断不等同精确源帧同步；文件视频与当前设备 IMU 未同步，不伪装为稳定证据。

## 资源与音频

`SpatialSessionAnalyzer` 用一个辅助 executor，在同一快照上串行运行 Depth、SegFormer；最多一项在途，无推理积压，模型 CPU intra/inter-op 各 1 线程。基础采样间隔 2 秒，按耗时三倍调整到最多 10 秒；主 YOLO 连续慢帧会降频，低内存/严重热状态暂停辅助推理。空间模型加载/推理失败分别标 UNKNOWN，继续保留独立 YOLO 主链。停止或关闭空间会话在其 worker 上释放模型。

Depth ONNX 约 99 MB。早期直接 `readBytes()` 在 AVD 192 MB Java heap 下触发 OOM；当前通过 32 KiB 流式复制/哈希解包到 `noBackupFilesDir`，再由 ORT 文件路径加载，修改后真实 Depth 同张量测试已 PASS。首次使用会增加本地模型副本磁盘空间。

`DirectionalAudio` 为左/中/右 stereo gain 与 TTS pan：左右侧主声道 1、另一声道默认 0.2，中间双声道 1；系统公告居中。沿用原提醒排队、去重、静音与有效期；这是三方向 panning，不是 HRTF、三维空间音频或距离模拟。动态环境概览缺少离线中文 TTS 时仅显示文字，风险仍可使用本地提示音。

## 复现与验证边界

在仓库根目录 PowerShell 执行；本机独立 `android/.export-venv` 的依赖见 `android/requirements-model-export.txt`，通过局部 PYTHONPATH 复用 PC 包，不改变 PC `uv.lock`。先按已固定来源及 SHA 准备本地三个 checkpoint；导出不更换 checkpoint。实际版本、加载信息、形状、哈希、相同输入数值对照和暖机记录保存在报告中。

```powershell
. ./android/tools/task_environment.ps1
$env:PYTHONPATH = (Resolve-Path .venv/Lib/site-packages).Path
& android/.export-venv/Scripts/python.exe android/tools/export_model.py
& android/.export-venv/Scripts/python.exe android/tools/export_segformer.py
Copy-Item android/spatial-models/segformer_mit_b0_ade20k_256.onnx android/app/src/main/assets/
& android/.export-venv/Scripts/python.exe android/tools/export_depth.py
# Depth 导出自动复制 main 模型并生成全面 manifest；仅刷新清单可加 --manifest-only。
Copy-Item android/parity-validation/v03-depth-python/depth_input.tensor.bin android/app/src/androidTest/assets/
Copy-Item android/parity-validation/v03-depth-python/depth_output.tensor.bin android/app/src/androidTest/assets/
Copy-Item android/parity-validation/v03-depth-python/depth_validation.json android/app/src/androidTest/assets/
./android/gradlew.bat -p android :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest --console=plain
```

导出后的 SHA 必须与当前 manifest/固定模型契约一致；软件变化导致不同 SHA 时须完成真实对照和设备验证后明确更新契约。模型、视频和生成验证二进制不提交 Git。前版 19 项 instrumentation 实际 PASS 是历史证据；v0.3 最终测试数量、APK SHA、延迟/内存及生命周期验收以此次独立 validation 文档和原始记录为准，不由历史结果推定。模拟规则测试、真实 Torch/ORT 推理和真实 Android 设备测试必须分别标注。
