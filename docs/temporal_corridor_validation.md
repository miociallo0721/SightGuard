# Stage 4A 阶段任务报告：真实图像运动与时间证据

日期：2026-10-11。八段 UrbanEgo 真实连续 RGB 共 2,400 帧已完成 CPU 处理，并关联原 240 个 Corridor 时刻、B/C/D 共 720 行。原模型未重新推理，所有原走廊成本/排序/状态保持，2,631 个冻结文件不变，独立重放原 257 条 AlertEvent 完全一致。

关键限制：222/240 窗口为 LIMITED、18/240 为 UNKNOWN。本配置没有可复用窗口，Freshness 全部拒绝，因此没有真实 FRESH/STALE 样本，不能声称 TTL 得到验证或系统已提升导航安全性。Temporal Stabilization 未运行。

## 数据、配置与溯源

同一次 UrbanEgo 原始拍摄的八段 15 秒片段，1280×720，发布视频 20 FPS；实际逐帧使用已验证 MP4 PTS，与 tracks 匹配，不把 FPS 当作时间真值或传感器 capture time。原始来源/许可、裁剪映射及视频哈希保留在 `datasets/real_world/manifests/dataset.json`；源传感器时间映射仍 unavailable。八段不视为八个独立随机拍摄样本。

输入：`outputs/spatial_grid/v03`、`outputs/corridor_intelligence/v01`、`outputs/real_world/v01/*/perception/tracks.jsonl`。上游 Depth v03、Path Relevance v04、GT Schema 3.0 保持冻结。配置/接口见 [实现说明](motion_quality_v1.md)。

- motion config SHA-256：`7d3e91cbcc71f23fb1994ec56177ad513e64c46923a79f2100420e168d00e88c`。
- uv.lock：`a95c90b4571d20df3642b6f8887f8c2ae7f4a00333ff787a6a5e6d96c2962637`。
- Spatial 元数据：`3f19d257f695007adbee6e1196989cfdf3355ecd1672015e26f25501f00050b1`。
- Corridor 元数据：`4d0cd18454a5d3db6d905694e230baf8800382b937e7a74705f44c572612330b`。
- 运行环境：Python 3.12.15、NumPy 2.5.3、OpenCV 4.14.0、Windows 11；i7-12650H，10 核/16 逻辑处理器，OpenCV 1 线程。没有加载 Torch 或新模型。

源码逐文件、输入逐视频/结果、输出逐文件哈希保存在运行元数据。`code_commit_before_run` 为上阶段 `be60e92290e0fbc7a67aee30240edfae52d10588`；新增实现运行时尚未提交，因此应以源码 SHA-256 为准，不能把该父提交当作本阶段实现提交。

输出版本：

- `outputs/motion_quality/v01`：遮罩/未遮罩各 2,400 条帧记录、240 条窗口、计时、每视频与整体 summary、失败索引、240 张真实 RGB 预览、八视频 HTML。
- `outputs/temporal_corridor/v01`：720 条原 B/C/D 的只读附加诊断、模拟 Freshness、summary、failure_cases、运行元数据。
- `outputs/motion_quality_validation/v01`：独立基线重放、验证 JSON、真实浏览器八张截图及 DOM 检查记录。
- `outputs/motion_quality_analysis/v02`：额外描述性关联、每视频两张失败/降级帧、16 帧真实 contact sheet。v01 是格式修正前的分析，保留未覆盖；v02 是报告采用版本。

## 帧级运动与遮罩对照

主遮罩估计：有效 2,308/2,400（96.17%），UNKNOWN 92（3.83%）。STABLE_IMAGE_MOTION 789（32.88%）、MODERATE 444（18.50%）、RAPID 13（0.54%）、LIMITED 1,062（44.25%）。这些是工程图像分类，不是现实相机稳定性或姿态真值。

每视频次序均为 stable / moderate / rapid / limited / unknown；各段 300 帧：

- V01：125 / 53 / 2 / 114 / 6；未遮罩 UNKNOWN 6；遮罩改变 35 个质量标签。
- V02：112 / 57 / 1 / 124 / 6；未遮罩 UNKNOWN 5；改变 27。
- V03：93 / 45 / 1 / 152 / 9；未遮罩 UNKNOWN 9；改变 30。
- V04：93 / 68 / 4 / 99 / 36；未遮罩 UNKNOWN 1；改变 99。
- V05：70 / 40 / 3 / 178 / 9；未遮罩 UNKNOWN 8；改变 30。
- V06：75 / 77 / 1 / 135 / 12；未遮罩 UNKNOWN 4；改变 42。
- V07：118 / 49 / 0 / 130 / 3；未遮罩 UNKNOWN 3；改变 28。
- V08：103 / 55 / 1 / 130 / 11；未遮罩 UNKNOWN 8；改变 20。

UNKNOWN 原因计数允许一帧多个原因：无前帧 8、可用背景面积不足 43、有效匹配不足 17、RANSAC consensus 不足 18、背景特征覆盖不足 7、变换超出二维模型范围 2。没有真实样本因初始特征点不足、解码失败或 PTS 错误失败；这些路径由自动化测试覆盖。除首帧外 84 帧 UNKNOWN，不宜统称为特征不足。

未遮罩有效 2,356（98.17%），UNKNOWN 44。遮罩共改变 311/2,400 个质量标签；51 帧从未遮罩有效变为遮罩 UNKNOWN，3 帧反向恢复。两版同时有效 2,305 帧，速率绝对差 P50 0.00276、P95 0.06177 diagonal/s。所有可用拟合诊断的主内点比例 P50 0.9184，未遮罩 0.8983；主内点残差 P95 的跨帧 P50 1.7949 work px。此统计包括随后被拒绝但已有数值的拟合，不能据此证明准确率更高。

## 时间窗口与候选变化

20 FPS 相邻帧估计映射到原 2 Hz 精确 key，窗口 `(t−500,t]`。一般 10 个间隔；初始窗口只有首帧、无有效运动，不填零。所有保存引用均通过无未来帧与同 stream 检查。

窗口标签按视频：V01 29 LIMITED/1 UNKNOWN，V02 29/1，V03 28/2，V04 23/7，V05 28/2，V06 27/3，V07 29/1，V08 29/1。18 个 UNKNOWN 包括 8 个起始窗口；222 个 LIMITED 大多含至少一帧拟合残差、覆盖或突变质量受限。任一 LIMITED 就限制完整窗口的规则很保守，尚未校准可用性。

每原组有 232 对视频内相邻样本，不跨视频比较。B/C 的最低成本候选交换为 89/92 次，best 身份含 null 的变化为 103/101 次；二者不是同一指标。B/C 分别 BEST→TIED 34/32、TIED→BEST 34/32、进入 ABSTAIN 4/4、ABSTAIN 恢复 3/3。近成本排序交换各 61；图像区域证据明显变化各 164，资格门禁变化各 61。分类可重叠，不能相加成为事件总数。

同名候选最大绝对成本项差值的描述：B 为 ground 306、obstacle 194、continuity 145、unknown 39、conflict 4；C 为 ground 300、obstacle 185、continuity 144、unknown 36、depth 25、conflict 6。全零增减不指定 dominant 项。这里统计的是所有候选的非零变化，不是“每次切换原因”的唯一计数。

## 质量、运动幅度和目标数量的关联

窗口 LIMITED 的 222 对中，B/C 最低成本交换 85/88（38.29%/39.64%）；UNKNOWN 的 10 对中各 4（40%）。B/C 状态变化 LIMITED 71/68、UNKNOWN 4/3。UNKNOWN 组很小；不能推断估计失败必然使评价更不确定。

以窗口中有效子集的速率 P95 为描述代理，≥0.5 diagonal/s 的 137 对中 B/C 最低成本交换 64/66（46.72%/48.18%）；低于阈值的 95 对为 25/26（26.32%/27.37%）。较大观测幅度伴随更多交换，但这些窗口本身仍 LIMITED/UNKNOWN；不是可信姿态、可校准运动条件或因果结论。

高置信检测至少 3 框的 124 对中 B/C 最低成本交换 43/45（34.68%/36.29%）；少于 3 框的 108 对为 46/47（42.59%/43.52%）。框数不等于遮挡面积/独立运动；此数据没有支持“目标越多就越容易切换”的简单结论。

Depth 引起原 B/C 完整排序差异仍为 9/240，全部处于 LIMITED；LIMITED 本已占全部采样 92.5%，所以不能宣称深度排序变化特别集中于某类高运动，更不能推断通行意义。B/C 状态差异 10 个样本，8 LIMITED、2 UNKNOWN。完整 key 保存在 temporal summary 的 b_c_differences。

## Freshness 与四组对照

研究组与原 Corridor A/B/C/D 分开命名：T0_BASELINE 和 T1_MOTION_DIAGNOSTIC 保持原 B/C/D 的每行源结果完全相等；T2_FRESHNESS_GATE 只增加研究门禁字段；T3_STABILIZATION 为 NOT_RUN。

五个模拟年龄每档：B/C 各 UNKNOWN 236、INVALID 4，FRESH/STALE 都为 0；原 D 每档 INVALID 240。共 3,600 条模拟 assessment：UNKNOWN 2,360、INVALID 1,240。T2 对 B/C 各把 236 个原非弃权结果研究性拒绝，使研究 ABSTAIN 240/240；原 B/C 实际仍各 4 ABSTAIN，原 D 仍 240 ABSTAIN。

哪些旧结果“过期”：本轮没有达到可信窗口前提的旧结果，因此没有被正式分类为 STALE 的真实记录。所有旧结果都因质量不足或原资格无效而拒绝，包括 age=1000 ms；不能把这称为已验证的过期检测成功。单位测试验证 FRESH、STALE、过期拒绝与非法消费时间，但真实 TTL 行为、误拒绝代价和及时失效敏感性仍未验证。

Temporal Shadow 没有保持候选、没有额外决策延迟实验；hold/interruption/switch reduction/extra delay 为 null，不用 0 伪装已运行。未使用过期证据；仍存在未来若盲目按来源窗口 TTL 保持结果而忽略新障碍的风险，需要当前证据、独立真值与跨帧对应验证。

## 真实失败样本与浏览器

实际检查八段 16 张真实失败/降级帧 contact sheet，不只是成功图；JPEG 从真实源视频解码，并校验帧 PTS：

- V01 f24：31 matches、约 45.16% inliers，但仅 14 内点，小于 30，consensus 不足。可见纹理地面和行人，纹理存在不保证可跟踪。
- V02 f74：60 matches、约 78.33% inliers，但匹配集中在局部，背景特征网格覆盖不足；f10 的可用面积仅 29.73%。
- V03 f7：车辆和人遮挡，consensus/覆盖不足；全段遮罩可用面积 P50 71.23%，UNKNOWN 9、LIMITED 152。没有断言车辆一定在独立运动。
- V04 f46：近墙与车辆填满画面，排除后可用面积 13.75%，UNKNOWN；f100 在真实 viewer 亦 UNKNOWN，过去窗口仅 40% 有效时间、300 ms 缺口。全段遮罩面积不足 35 帧；语义不确定与背景缺少不能靠 UNKNOWN=静止化解。
- V05 f8：7 个有效匹配，UNKNOWN；晚段有椅子，conf≥0.5 的 chair 帧检测实例累计 221（不是 221 个独立椅子），全部也被排除。全段可用面积 P50 83.46%，遮罩增加 1 个 UNKNOWN 净差，未建立静态/动态物体真值，不能把静态特征误遮挡代价量化为准确率。
- V06 f47：68 matches、75% inliers，但空间覆盖不足。数量/比例本身不能覆盖质量要求。
- V07 f80：遮阳棚、行人、桌椅场景，102 matches、29.41% inliers、变换参数超界；UNKNOWN。f100 实际 viewer 显示帧和窗口 LIMITED、原 C BEST_SUPPORTED、研究复用拒绝。未遮罩/遮罩各 UNKNOWN 3，并不能证明独立运动已被解决。
- V08 f55：重复纹理地面，66 matches、27.27% inliers，consensus/覆盖不足；图像方向变化不提供绝对俯仰。D 全弃权。f100 主帧可拟合 moderate，但窗口 LIMITED。

真实浏览器通过 `http://127.0.0.1:8767` 检查全部八段 f100、查看 C 的成本/状态、质量和模拟 Freshness，并检查未遮罩、age=1000 ms、原 D 显示 INVALID/ABSTAIN；控制台没有 error。截图及 DOM 在 validation/browser。旧 8766 viewer 和用户 7860/7861 服务未停止。合成图和 mock 只用于自动化故障/数值测试，不冒充真实浏览器或真实视频实验。

## 性能

一次批量处理耗时 104.7895 s，从父输入 preflight 成功、创建新输出后计时，包含视频处理、两版运动、诊断、预览、汇总、输出写入和哈希；不含此前 preflight 和最后两份根元数据写入。未记录启动至退出的完整 CLI 耗时，不能将 104.79 s 称为包含所有预检的端到端延迟。独立基线回归、补充分析、浏览器抽样另行执行。

纯主运动按逐帧总耗时为 52.79 FPS；解码+主运动 47.93 FPS；完整双对照研究流程平均 22.90 source frames/s。进程 working set 393.85 MiB、peak working set 485.56 MiB。GPU 未使用；这不是包括 YOLO/SegFormer/Depth、正式提醒或 live UI 的系统实时 20 FPS 证明，也没有在线 deadline/jitter 验证。

阶段 P50 / P95（ms；缺少阶段不会补零，样本数不同）：

- 解码 1.8486 / 2.3467（2,400）。
- 准备/遮罩 4.9603 / 6.0170（2,392）。
- 特征提取 3.5247 / 4.3588（2,349）。
- 双向光流/匹配过滤 9.7921 / 15.7553（2,349）。
- RANSAC 全局变换调用 0.0784 / 0.1046（2,332）。
- quality 最终分类 0.0203 / 0.0259（2,330）；未单独计入前面的残差/诊断整理，纯主运动总计覆盖它们。
- 纯主运动 19.2256 / 25.4634；未遮罩总运动 19.3216 / 26.1069（各 2,400）。
- 解码+主运动 21.0362 / 27.4206（2,400）。
- 窗口聚合 0.2621 / 0.3586；走廊关联/诊断 0.2610 / 0.8573（各 240）。
- 两份帧级 JSONL 序列化 0.3840 / 0.6060（2,400）；其余汇总与 HTML 写入计入整体 wall，未单独做逐帧分位数。
- RGB 预览与 viewer 样本准备 3.1397 / 3.7654（240）；浏览器渲染耗时未测。

## 测试、回归与复现

默认全仓库：559 passed、27 skipped；新增 motion 核心/契约 51 项通过，其中 4 项明确 mock 故障注入，真实 OpenCV 合成图检验平移大小、动态 patch 排除和确定性，其他测试覆盖时间、输入、窗口、Freshness、状态分类、隔离等。默认不要求数据下载/GPU。

另行开启真实视频集成：9 passed，包含八段各前 12 帧重新解码/实际 OpenCV 与保存估计比较，以及独立基线验证证据。完整真实批量则处理了每段全部 300 帧；不是仅 96 帧小样本推导总数。Ruff check 全仓库通过，format 全仓库通过。

独立验证逐行比较原 Corridor B/C/D 与新 source_corridor、两版 frame key、past window 引用、姿态未知及 Freshness 拒绝；核验全部新输出哈希与源码哈希。基线重新运行 assess_tracks，比较 2,400 条 risk_frames、events、alerts 原 JSONL，257 条 AlertEvent 全相等；2,631 个冻结文件无差异。没有调用音频适配器。

以下命令在仓库根目录执行。输出目录必须不存在；示例 new-run 应按本地实际版本更换，不覆盖 v01：

```powershell
uv run --no-sync python scripts/run_motion_quality.py --motion-output outputs/motion_quality/new-run --temporal-output outputs/temporal_corridor/new-run
uv run --no-sync python scripts/validate_motion_quality.py --motion outputs/motion_quality/new-run --temporal outputs/temporal_corridor/new-run --output outputs/motion_quality_validation/new-run
uv run --no-sync python scripts/analyze_motion_quality.py --motion outputs/motion_quality/new-run --temporal outputs/temporal_corridor/new-run --output outputs/motion_quality_analysis/new-run
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
$env:SIGHTGUARD_MOTION_INTEGRATION='1'
uv run --no-sync pytest -q tests/test_motion_quality_integration.py
Remove-Item Env:SIGHTGUARD_MOTION_INTEGRATION
uv run --no-sync python -m http.server 8767 --bind 127.0.0.1 --directory outputs/motion_quality/v01
```

opt-in 集成当前明确检查本次 v01 与 validation/v01；验证其他版本时需指定对应测试 fixture/路径，不能把 v01 验证冒充 new-run。HTTP 端口若已有本阶段服务，无需重复启动；不能为端口冲突强制停止用户服务。

## 阶段验收与 Git

最终请求 18 项均在本报告对应说明：实际算法、Roana 边界、能/不能估计、分布、遮罩、UNKNOWN、时间对齐、分类、质量关联、Depth、Freshness、Shadow、D、257 条回归、性能、测试、提交及下一阶段证据。

提交分支 `codex/motion-quality-v1`；基于 Corridor 提交 `be60e92290e0fbc7a67aee30240edfae52d10588`，不合并 master，不 push。实现与本报告在同一次提交；提交完成后精确 ID 写入本地 `outputs/motion_quality_validation/v01/git_receipt.json` 和包含 ID 的 `stage4a_task_report.md`，最终答复也提供该 ID。报告自身不能嵌入包含自身的 Git commit hash；可用 `git log -1 --format=%H -- docs/temporal_corridor_validation.md` 查到对应提交。

下一阶段仍缺 IMU/可靠标定与姿态真值、静态背景/独立物体运动真值、独立时序通行/危险变化标注、跨帧候选几何对应、重复图像与真实运动的区分、TTL 与 LIMITED 规则的敏感性/误拒绝校准、不同拍摄来源和长序列验证、当前新障碍失效机制及真正在线端到端延迟测量。本轮只交付诊断接口，不进入正式提醒仲裁、导航状态机或动态 HRTF。
