> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Stage 4A 暂停进度

历史快照：用户于 2026-10-11 要求暂停时保存。随后已获用户「继续任务」指示并恢复开发；本文保留暂停时状态，最终完成情况见 [阶段报告](temporal_corridor_validation.md)，不能把以下历史待办视为当前未完成项。

## 工作树与任务来源

- 仓库：`D:\program\SightGuard`。
- 当前分支：`codex/motion-quality-v1`。
- HEAD：`be60e92290e0fbc7a67aee30240edfae52d10588`（上一阶段 Corridor Intelligence 提交）。
- 本阶段请求：`C:\Users\55043\.codex\attachments\b3cc2a05-e826-4b64-83a4-bf6c0147a190\已粘贴的文本.txt`；恢复时应重新读取。
- 本阶段代码尚未提交、尚未 push。所有新增文件保留在工作树。
- `docs/architecture_consolidation_task_report_2026-10-11.md` 是任务开始前已存在的未跟踪文件，不能混入本阶段提交。
- 已执行一次 `git fetch --no-tags origin`，未合并；origin/master 更新至 012ec82，本分支仍从上述 Corridor 提交创建。
- 本阶段最终要求：仅暂存本任务文件，以 `experiment: add camera motion quality and temporal corridor diagnostics` 提交；不自动 push，不合并 master，不进入下一阶段。

## 已完成

1. 审计项目规则、原有接口、Corridor/Spatial 元数据与八段视频输入。
2. 冻结 2,631 个原有文件的 SHA-256，保存在 `outputs/motion_quality_audit/v01/frozen_hashes.json`；创建时逐项验证通过。包含原有模型输出、Corridor v01、已有源码/配置等。本阶段新增文件不在冻结集合内。
3. 八段源视频均存在，视频哈希与 manifest 的 processed_sha256 一致。每段跟踪结果 300 帧，视频 PTS，20 FPS，约 15 秒。
4. 新增配置 `configs/motion_quality.yaml`。
5. 新增 `src/sightguard/scene/motion/`：
   - `schema.py`：配置、MotionFrame、帧级估计、质量、窗口数据结构与版本信息。
   - `estimator.py`：Shi–Tomasi 特征、双向 LK、前后向一致性过滤、RANSAC partial affine；保留二维平移/旋转/尺度代理与残差。可靠检测框加边缘排除；另支持未遮罩对照。
   - `temporal.py`：仅过去 500 ms 的窗口聚合、模拟消费时间 Freshness、原有 Corridor 状态变化分类和成本项差值。
   - `__init__.py`。
6. 新增 `tests/test_motion_quality.py`，44 个参数化测试通过；包含真实 OpenCV 合成图测试和明确命名的 mock 失败测试。
7. 新文件 Ruff 检查与格式检查曾通过；之后仅将一个测试名称改为英文，尚未再跑最终完整检查。

## 已有验证及其范围

- 修改前默认测试：508 passed、18 skipped。
- 修改前保存结果集成检查：8 passed、1 skipped。
- 新增核心测试：44 passed。不是全仓库修改后最终验收。
- 真实视频小样本：`urbanego_run8_07` 连续解码前 12 帧，解码 PTS 与跟踪 PTS 差值小于 0.01 ms；执行真实 OpenCV 遮罩估计。质量分布：7 STABLE_IMAGE_MOTION、1 MODERATE_IMAGE_MOTION、3 LIMITED、1 UNKNOWN（首帧无前帧）。此小样本未传入 previous_estimate，因此未检验连续估计突变规则，不能替代全批次。
- 小样本呈现部分相邻图像运动极低、部分帧跳变。尚未验证是否源视频重复图像，不得解释成真实相机静止。
- 原八段 2,400 帧、257 条 AlertEvent 的基线事实已审计；本阶段新增代码后的独立重放回归尚未执行。

## 算法与边界

- 质量：STABLE_IMAGE_MOTION / MODERATE_IMAGE_MOTION / RAPID_IMAGE_MOTION / LIMITED / UNKNOWN，综合特征数、匹配、内点比例、背景覆盖、残差、模糊、真实 dt 与突变。
- 特征不足、无前帧、解码失败、跨序列、帧序/时间错误、过大间隔、RANSAC 失败或不可信模型必须 UNKNOWN；不得零填缺失运动。
- 所有 absolute_camera_pose_status 保持 UNKNOWN；只测二维图像运动，不能估计绝对俯仰、三维姿态、人体速度或碰撞预测。不得解除原严格 D 组姿态门禁。
- 可靠检测框包含可能静止的椅子；排除框并不证明剩余像素静止，需真实对照分析。
- Freshness 的消费时刻是模拟的视频时间，不能称为在线或音频延迟。UNKNOWN/LIMITED 不允许研究复用；旧 Corridor ABSTAIN 为 INVALID。
- 可选 Temporal Shadow 暂不实现：缺少可靠的跨帧候选对应及独立真值；配置开启会明确拒绝。最终需报告 NOT_RUN 和理由，不能报告虚假平滑收益。
- 没有改动 risk/decision/正式 AlertEvent/HRTF/live 默认行为，没有调用生产音频。

## 外部参考已核验

- Roana 本地审计副本：`D:\program\sightguard-audit-roana-20261011`，固定 revision `7da28b7dd127f002986711ea62249b73de723739`。
- 阅读 MotionQuality.kt、CorridorStateMachine.kt、CorridorPipeline.kt、MainActivity.kt 的帧时间相关实现。借鉴质量/新鲜度/失效分类概念，不复制 AGPL 源码或测试。
- Roana 缺失运动返回 STABLE 的规则不能继承；其 IMU pitch/angular velocity 阈值不能用于本项目无 IMU 的二维图像估计。其 CorridorPipeline 未接入实测 MotionQuality，也未提供量化 TTL。
- 官方算法文档：<https://docs.opencv.org/4.x/dc/d6b/group__video__track.html>、<https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>。

## 恢复后的待办（尚未实施）

1. 先核验工作树及请求；保持暂停，只有用户要求继续才恢复开发。
2. 可修正 `diagnose_change` 的 dominant cost 项：所有差值为零时应返回 null，而不是任选第一项。检查窗口 valid estimate 的前帧身份和 dt 一致性是否需要加强，并添加有效测试。
3. 新建 CPU 离线 runner `src/sightguard/pipeline/motion_quality.py` 和 CLI；复用 `pipeline.corridor.preflight` 的原 Spatial 输入校验，校验原 Corridor 文件与源码哈希。
4. 按真实解码 PTS 逐帧处理全部 8×300 帧，同时输出遮罩与未遮罩对照。PTS/解码异常保留 UNKNOWN 和错误，不使用假定 20 FPS 修复时间。
5. 生成新版本目录 `outputs/motion_quality/v01` 与 `outputs/temporal_corridor/v01`（暂停时尚未创建）。禁止覆盖现有目录。保存 frame_motion.jsonl、unmasked_frame_motion.jsonl、motion_windows.jsonl、temporal_corridor.jsonl、metadata/summary/failure_cases/timings/hash 与预览。
6. 将因果 500 ms 运动窗口严格对齐原 2 Hz 的 240 个 Spatial/Corridor frame key。原 B/C/D 共 720 行只读分析，保留原成本、排序、状态；区分最低成本候选变化、BEST→TIED、TIED→BEST、ABSTAIN、恢复、近成本排序交换与区域证据变化。null 身份变化不等同方向抖动。
7. 研究对照组拟定 T0_BASELINE、T1_MOTION_DIAGNOSTIC、T2_FRESHNESS_GATE、T3_STABILIZATION_NOT_RUN；不得混淆原 Corridor 的 A/B/C/D。T0/T1 原结果相同，T2 只新增研究字段，T3 指标 null/NOT_RUN。
8. 模拟消费年龄 0/100/250/500/1000 ms，按配置 TTL 分析 FRESH/STALE/INVALID/UNKNOWN；原 D 仍全部 ABSTAIN。
9. 分析实际运动质量、遮罩影响、UNKNOWN 原因、运动质量与变化的关联（不能声称因果）、B→C 的 depth 微小排序变化、失败案例，尤其 V03/V04/V05/V07/V08。
10. 真实 RGB 预览与只读 HTML viewer；展示运动点/线、遮罩、原网格/走廊、质量、Freshness、来源和原因。不能有导航箭头。真实查看全部八段与失败样本，明确合成测试与真实视频区别。
11. 补充 runner/配置/失败/保存结果测试与独立 opt-in 真实视频集成检查。
12. 使用冻结集合验证全部 2,631 文件未变，独立重放八段原风险/事件/提醒，证明 257 条 AlertEvent 完全一致。既有 scripts/validate_corridor.py 和 validate_spatial_grid.py 可参考或复用。
13. 真实计量 decode、features、LK、global estimate、quality、window、join、serialization、visualization 的 P50/P95 与进程内存；区分纯运动、含解码、完整研究流程。不能由单模块速度宣称整系统实时 20 FPS。
14. 新增 `docs/research/motion_quality_v1.md`、`docs/archive/temporal_corridor_validation.md`，必要时更新 README；回答请求末尾全部 18 项。
15. 运行 `uv run --no-sync ruff check .`、`uv run --no-sync ruff format --check .`、`uv run --no-sync pytest -q` 及真实集成；检查 Git diff 后按请求提交，不 push。

## 保持现有资源

- 用户服务和旧 viewer 未停止：检查时端口 7860（PID 32332）、7861（PID 2220）、8766（PID 29056）。恢复时 PID 可能变化，不能据此强制终止。
- 旧 viewer 地址：`http://127.0.0.1:8766/urbanego_run8_07/index.html`；这是上一阶段结果，不是 Stage 4A 新可视化。
- 不修改 AGENTS.md、全局配置、旧冻结源码/数据或用户报告；不删除研究文件，不强制 Git 重置。
