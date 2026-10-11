# 单目米制墙距与局部图像候选研究

本轮交付是默认关闭的研究原型。已有墙面存在提醒、YOLO 风险与中文离线音频继续使用 0.4 链路；没有新增距离、接近或通行自动语音。

## 基线与隔离

- Android 0.4 基线提交：`4a1fdfb97190a9a85e151479ec19a13b8c37bc73`。
- 先前修改已提交 [PR #6](https://github.com/miociallo0721/SightGuard/pull/6)，GitHub 显示已合并，merge commit `0f142a985068b4d11d04ceeeb7a04706eff33472`。
- 本轮独立分支：`codex/metric-depth-corridor-research`。没有合并 master 或推送研究代码。
- 稳定 APK SHA256：`6671a81d740d5a12087f7af6eb8796b7565f8c28add75f59ffbcc5f022753fd7`。
- `android/deliveries/metric-corridor-research/baseline/` 保存 APK、源文件快照、初始 Git 状态与旧验证记录。主目录的 PC/UI 未提交改动和其他 worktree 保留。
- 本轮没有修改冻结的 PC Grid/Corridor、正式 RiskScorer、AssessmentEngine、AlertScheduler 或 SpeechController。

## 模块与数据流

`src/sightguard/research/wall_metric.py` 使用真实 SegFormer class 0 的置信度、连通区域、中央 ROI、YOLO 前景排除、同帧键与有效期，统计有效预测 Z 的 median/P10/P90。无效来源、过期、混合平面、有效覆盖不足明确弃权。P10/P90 是空间像素分布，不能充当准确率区间。

`wall_approach.py` 使用墙面掩膜关联、Shi–Tomasi + 双向 LK、RANSAC 相似变换、特征覆盖、标定射线的 Wahba 纯旋转反模型和连续窗口。相对深度变化或米制预测波动都不能直接触发接近。输出只涉及相机运动候选，身体运动未验证。没有可信 IMU 同步时标记 `MOTION_SYNC_UNKNOWN`。

`local_corridor.py` 复用原 15×15 Spatial Grid 和 CorridorEvaluator 的 B/C 图像证据比较，保留原严格 D 组门禁。增加未知、冲突、墙面、检测框覆盖、时效和稳定性检查；未知格不是空闲格。输出范围始终 `CANDIDATE_ONLY`，物理宽度、身体朝向和净空未知。

Android `MetricDepthMath.kt`、`MetricDepthDetector.kt`、`WallMetricResearch.kt` 是独立适配器。官方等比例 lower-bound resize、多 14 输入、cubic RGB/ImageNet 预处理，四个静态图共享一份权重，按形状仅保留一个 CPU ORT session。MainActivity 只增加开发者开关和最近研究记录；局部 Corridor 仍只在 PC 运行。

SpatialSessionAnalyzer 继续只有一个辅助 worker、至多一份源快照。语义结果在米制推理前交回正式环境链；米制采样间隔 12 秒，与原 Depth 同次互斥。资源不足、主链慢或热状态升高时不运行。米制模块关闭时不加载其 ORT session。

## 时间契约

PC 使用 TUM 原始 RGB 时间戳；原运动模块不支持这些稀疏 dataset 时间源时继续 UNKNOWN，没有伪装成连续 video PTS。离线“源年龄”是实测模型处理时间组合的模拟串行年龄，不能冒充 CameraX 实测帧龄。

Android 保存 capture timestamp、时间源、接收时间、完成时间、FrameKey、尺寸和旋转。若 Camera2 声明 `SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME`，新墙距模块直接检查 sensor capture 与 elapsedRealtime 的年龄；否则只记录 analyzer 接收/文件解码请求后的年龄，并明确不能保证曝光年龄。没有任意固定偏移补偿。TTL 仍为 500ms，不因辅助模型缓慢而延长。

当前设备此前报告 Camera2 时间源 UNKNOWN；本轮新版未能安装，因此没有新的时间同步测量。当前设备快速转动只能作为独立拒绝信号，不能表述为精确跨源融合。Android 官方 [timestamp source 契约](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_TIMESTAMP_SOURCE) 支持这一保守处理。

## 来源、模型与许可

- [Depth Anything V2 官方 Metric Depth](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md)，源码固定 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`。
- [Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small)，revision `3bc65d4e14a6786a61acec16453c50e12bf5f338`，SHA256 `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`。
- [VKITTI Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small)，revision `c725b8589bdf6ab04072cab74c0467830db80d6d`，SHA256 `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`。
- 两个 Small 权重 Apache-2.0，max_depth 分别 20/80m。官方 Hypersim loader 的 `hypersim_distance_to_depth` 将射线距离转换为光轴 Z；本轮不把 Z 当作任意方向欧氏距离。
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)，CC BY 4.0。引用 Sturm, Engelhard, Endres, Burgard, Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012。
- [TUM 格式与标定](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)：registered depth uint16/5000，0 无效，不重复施加已有深度标定因子。

官方文件通过正常 TLS 下载，没有关闭证书检查。TUM 未获取到发布方校验和；报告中的 archive SHA256 是本地计算的复现标识，不能表述为核对官方 checksum。

## 已验证与限制

5 个室内序列共 700 个冻结均匀样本，执行真实 SegFormer、YOLO、原相对深度，以及 Hypersim 518/280、VKITTI 518 共 2100 次米制推理。每个输入有哈希、深度/pose 关联、模型版本和失败记录。没有语义墙面真值，深度误差以“预测墙面区域”定义，不是墙面检测准确率。没有室外 RGB-D 真值评测，也没有手机卷尺距离验证。

Hypersim 280 的预测墙区域 MAE 依序列约为 1.8440、1.2922、0.3474、0.4372、0.1106m，差异明显；不能只引用最好序列宣布测距精度。完整 MAE/RMSE/AbsRel、像素加权分位数和距离分组在交付 CSV/JSON。经验误差带约 ±2.1129m，拒绝作为已校准距离区间。

接近研究 700 帧全部弃权。仅 2 对独立 mocap 明确前移标签且没有确认墙面，没有有效后退标签，真实正例不足，不能宣称已经区分前移与旋转。合成扩张成功仅验证几何核规则。局部候选 24 条稳定研究记录、13 条并列；严格 D 组 700 帧全部弃权，没有 SAFE/PASSABLE 或路线建议。

逐一复核 24 条支持记录，全部集中在同一贴海报的平面场景，候选覆盖海报/板面，无法确认实际行走地面。官方描述该序列为有纹理平面，未提供通行地面标签。这些记录不能算作“真实局部地面候选已验证”，也不据此计算假正率；失败图和全部原始结果保留。低纹理序列同样只确认 RGB-D 平面深度与模型预测 wall，不能冒充独立墙面语义真值。

工程验证：614 项 PC 测试通过、27 项原有 opt-in/环境条件跳过，新增研究测试 55 项；Ruff 通过。Android JVM 最终数量见交付 `validation_summary.json`。四个 ONNX 形状分别在 PC ORT 1.31.0 和与 Android 同版本的 1.24.3 通过同输入 PyTorch 对照，最大差 <0.00005m。Kotlin cubic 边界/过冲的独立 OpenCV 参考检查通过。

新版主 APK 和 instrumentation APK 可构建；nova12 安装时要求锁屏人工确认，已撤销会话 765964172，没有操作锁屏或绕过授权。新版设备推理、CameraX、内存/FPS、生命周期及开发者界面未验收。已安装旧 0.4 本轮单独通过真实 SegFormer、相对深度和 15 句环境音频真实 onStart/completed 三项测试。不得把这些旧版测试等同新版验收。

## 复现

以下命令在研究 worktree 根执行。数据、权重、ONNX、APK、日志和图不提交 Git。外部缓存可由 `SIGHTGUARD_METRIC_CACHE` 指定。

实验使用独立 GPU 环境，其核心版本记录在 `research/metric-corridor/pyproject.toml` 和 `uv.lock`；完整已用运行环境另存交付版本快照。为节约磁盘，本轮没有再下载一套 GPU 环境重跑，不能声称新建锁定 GPU 环境已经单独验收。PC 同版本 Android ORT 的小环境实际由其嵌套 `uv.lock` 创建并运行过。

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) src)
$env:SIGHTGUARD_METRIC_CACHE='D:/program/SightGuard/models/metric-corridor'
uv sync --frozen --project research/metric-corridor
$metricPy=(Resolve-Path research/metric-corridor/.venv/Scripts/python.exe).Path
```

默认截止时间是本轮绝对 UTC 截止，以后复现须明确另设 `SIGHTGUARD_RESEARCH_FREEZE_UTC`、`SIGHTGUARD_RESEARCH_STOP_UTC`，并复制实验 YAML 后调整 `resources.operations_stop_utc`。本轮执行没有延后截止。不要对历史报告文件原地篡改实验配置。

先按 Android 0.4 文档准备已有三个模型。官方米制源码需 clone 到缓存下 `Depth-Anything-V2` 并 checkout 上述固定 SHA，再执行：

```powershell
& $metricPy scripts/fetch_metric_research.py --sequences freiburg1_xyz freiburg1_rpy freiburg3_nostructure_notexture_far freiburg3_structure_texture_far freiburg3_nostructure_texture_near_withloop
& $metricPy scripts/prepare_metric_dataset.py
& $metricPy scripts/prepare_metric_dataset.py --sequences freiburg3_structure_texture_far --manifest "$env:SIGHTGUARD_METRIC_CACHE/extension-manifest.json"
& $metricPy scripts/prepare_metric_dataset.py --sequences freiburg3_nostructure_texture_near_withloop --count 300 --append --manifest "$env:SIGHTGUARD_METRIC_CACHE/extension-manifest.json"
& $metricPy scripts/run_metric_research.py --phase semantics
& $metricPy scripts/run_metric_research.py --phase hypersim --input-size 518
& $metricPy scripts/run_metric_research.py --phase hypersim --input-size 280
& $metricPy scripts/run_metric_research.py --phase vkitti --input-size 518
& $metricPy scripts/run_metric_research.py --phase relative
& $metricPy scripts/run_spatial_research.py
```

扩展 400 帧使用同样的 5 个模型 phase，再运行 spatial；传入 `--manifest <extension-manifest.json>` 和 `--output android/deliveries/metric-corridor-research/evaluation-extension`，spatial 对应参数为 `--evaluation`。如使用以后创建的新配置，每条 model/spatial 命令需传 `--config <new-config.yaml>`。所有推理阶段串行运行。

ONNX 导出每次一个进程：

```powershell
$publicImage=(Get-Content "$env:SIGHTGUARD_METRIC_CACHE/sample-manifest.json" -Raw | ConvertFrom-Json).samples[0].rgb_path
foreach($shape in @(@(280,378),@(378,280),@(280,504),@(504,280))) {
  & $metricPy scripts/export_metric_depth.py --phase export --height $shape[0] --width $shape[1] --image $publicImage
}
& $metricPy scripts/export_metric_depth.py --phase pack
& $metricPy scripts/export_metric_depth.py --phase parity
Copy-Item "$env:SIGHTGUARD_METRIC_CACHE/Depth-Anything-V2/LICENSE" android/app/src/main/assets/metric_research/LICENSE.txt
& $metricPy scripts/prepare_android_metric_fixtures.py
uv sync --frozen --project research/metric-corridor/android-ort
& research/metric-corridor/android-ort/.venv/Scripts/python.exe scripts/export_metric_depth.py --phase parity
```

导出的四个模型诊断图片为生成目标比例而做 resize，仅用于同输入转换/预处理对照；它们不计入距离准确率。真实距离评估使用完整原始比例 TUM 图像。

```powershell
. ./android/tools/task_environment.ps1
Push-Location android
./gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest --offline --no-daemon --max-workers=1
Pop-Location
```

手机安装需要正常系统授权，本轮未完成，不能静默绕过。获正常授权后才可运行新增测试：

```powershell
adb -s <authorized-device> install -r -t android/app/build/outputs/apk/debug/app-debug.apk
adb -s <authorized-device> install -r -t android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s <authorized-device> shell am instrument -w -e class org.sightguard.app.MetricResearchInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

PC 汇总使用 `scripts/summarize_metric_research.py`。`build_metric_research_report.py` 仅从本轮冻结记录重建交付报告；其中旧 nova12 验收说明不能移用为其他设备结论。

完整 pytest 需要冻结合成夹具的原始 LF 字节；本轮恢复为 Git blob 并核对已有 manifest 哈希，新增 `.gitattributes` 仅固定该目录 LF，没有改标签/哈希。`test_vision_investigation` 还依赖既有公开 laya 源仓库的 example.jpg，本轮从主工作区复用同哈希文件，未下载或上传私有数据。默认测试不运行联网/GPU 模型。

## 资源与回退

PC 模型重任务串行、PyTorch/ORT 两线程、GPU 份额 0.65、采样点至少 1.5GB 可用 RAM、磁盘保留 5GB、下载限 8M；Gradle 单 worker、1.5GB heap、不启动 AVD。只降低自身研究进程优先级，不终止用户应用。资源采样不是全系统峰值监控，不能保证所有场景绝不 OOM。

稳定/实验 APK 使用相同 debug 证书；正式 APK 没有新增 INTERNET 权限。关闭开发者米制开关或安装保留的稳定 APK 可回退。详细路径、SHA256、安装授权限制、失败样本和未完成门禁见交付 `rollback.md`、`failure_cases.md`、`final_report.md`。
